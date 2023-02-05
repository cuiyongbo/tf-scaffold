// A minimal but useful C++ example showing how to load an mnist object
// recognition TensorFlow model, prepare input images for it, run them through the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones are supported.

#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"


static tensorflow::Status ReadEntireFile(tensorflow::Env* env, const std::string& filename,
                             tensorflow::Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  std::string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<tensorflow::tstring>()() = tensorflow::tstring(data);
  return tensorflow::OkStatus();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
tensorflow::Status ReadTensorFromImageFile(const std::string& file_name, std::vector<tensorflow::Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();

  std::string input_name = "file_reader";
  std::string output_name = "normalized";

  // read file_name into a tensor named input
  tensorflow::Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader = tensorflow::ops::Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
    {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 1;
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = tensorflow::ops::DecodePng(root.WithOpName("png_reader"), file_reader,
                             tensorflow::ops::DecodePng::Channels(wanted_channels));
  } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        tensorflow::ops::Squeeze(root.WithOpName("squeeze_first_dim"),
                tensorflow::ops::DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } else if (tensorflow::str_util::EndsWith(file_name, ".bmp")) {
    image_reader = tensorflow::ops::DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = tensorflow::ops::DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              tensorflow::ops::DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster = tensorflow::ops::Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = tensorflow::ops::ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = tensorflow::ops::ResizeBilinear(
      root, dims_expander,
      tensorflow::ops::Const(root.WithOpName("size"), {28, 28}));
  auto reshape = tensorflow::ops::Reshape(root, resized, {-1, 28*28});
  tensorflow::ops::Div(root.WithOpName(output_name), reshape, {(float)255.0});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
  return tensorflow::OkStatus();
}

tensorflow::Status GetPredictedLabel(const std::vector<tensorflow::Tensor>& outputs, int& predicted_label, bool verbose) {
  auto root = tensorflow::Scope::NewRootScope();
  std::string output_name = "arg_max";
  tensorflow::ops::ArgMax(root.WithOpName(output_name), outputs[0], 1);
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  std::vector<tensorflow::Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name+":0"}, {}, &out_tensors));
  predicted_label = out_tensors[0].vec<tensorflow::int64>()(0);
  if (verbose) {
    for (const auto& record : out_tensors) {
      LOG(INFO) << record.DebugString();
      LOG(INFO) << "DataType: " << tensorflow::DataTypeString(record.dtype()) << ", NumElements: " << record.NumElements();
      int dims = record.dims();
      LOG(INFO) << "dimensions: " << dims;
      for (int i=0; i<dims; ++i) {
        LOG(INFO) << "dimensions[" << i << "]=" << record.dim_size(i);
      }
      LOG(INFO) << "tensor values: " << out_tensors[0].vec<tensorflow::int64>()(0);
    }
  }
  return tensorflow::OkStatus();
}

/* Graph structure:

# saved_model_cli show --dir data/mnist/1 --tag_set serve
The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:
SignatureDef key: "__saved_model_init_op"
SignatureDef key: "serving_default"

# saved_model_cli show --dir data/mnist/1 --tag_set serve --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
  inputs['dense_2_input'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 784)
      name: serving_default_dense_2_input:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['dense_3'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 10)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict

# run the binary with:  
# ./mnist_demo_main --image=data/mnist_test_demo.jpg --graph=data/mnist/1 --input_layer=serving_default_dense_2_input:0 --output_layer=StatefulPartitionedCall:0
*/

int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than mnist, then you'll need to update these.
  std::string image = "data/mnist_demo.jpg";
  std::string graph = "./data/mnist_cpu";
  std::string input_layer = "serving_default_dense_input";
  std::string output_layer = "StatefulPartitionedCall";
  std::string root_dir = "";
  bool verbose = false;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("image", &image, "image to be processed"),
      tensorflow::Flag("graph", &graph, "graph to be executed"),
      tensorflow::Flag("input_layer", &input_layer, "name of input layer"),
      tensorflow::Flag("output_layer", &output_layer, "name of output layer"),
      tensorflow::Flag("root_dir", &root_dir, "interpret image and graph file names relative to this directory"),
      tensorflow::Flag("verbose", &verbose, "print more details about inference"),
  };
  std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<tensorflow::Tensor> resized_tensors;
  std::string image_path = tensorflow::io::JoinPath(root_dir, image);
  tensorflow::Status read_tensor_status = ReadTensorFromImageFile(image_path, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  const tensorflow::Tensor& resized_tensor = resized_tensors[0];

  tensorflow::SavedModelBundle model;
  tensorflow::RunOptions run_options;
  tensorflow::SessionOptions session_options;
  std::string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  tensorflow::Status status = tensorflow::LoadSavedModel(
    session_options, 
    run_options, 
    graph_path, 
    {tensorflow::kSavedModelTagServe}, 
    &model);

  // Actually run the image through the model.
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status run_status = model.GetSession()->Run({{input_layer, resized_tensor}}, {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  if (verbose) {
    // Do something interesting with the results we've generated.
    for (const auto& record : outputs) {
      LOG(INFO) << record.DebugString();
      LOG(INFO) << "DataType: " << tensorflow::DataTypeString(record.dtype()) << ", NumElements: " << record.NumElements();
      int dims = record.dims();
      LOG(INFO) << "dimensions: " << dims;
      for (int i=0; i<dims; ++i) {
        LOG(INFO) << "dimensions[" << i << "]=" << record.dim_size(i);
      }
      LOG(INFO) << "tensor values: ";
      // outputs[0] is a 1x10 matrix
      auto tm = outputs[0].matrix<float>();
      for (int i=0; i<outputs[0].dim_size(0); ++i) {
        for (int j=0; j<outputs[0].dim_size(1); ++j) {
          LOG(INFO) << "value[" << i << "," << j << "]=" << tm(i, j);
        }
      }
    }
  }

  int predicted_label = -1;
  run_status = GetPredictedLabel(outputs, predicted_label, verbose);
  if (!run_status.ok()) {
    LOG(ERROR) << "GetPredictedLabel failed: " << run_status;
    return -1;
  }
  LOG(INFO) << "predicted result: " << predicted_label;
  return 0;
}
