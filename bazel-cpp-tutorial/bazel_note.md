# Bazel FAQs

#. list all build targets

```bash
# bazel query //...
//:pip_pkg
//tensorflow_model_optimization:build_docs
//tensorflow_model_optimization:tensorflow_model_optimization
//tensorflow_model_optimization/python:python
//tensorflow_model_optimization/python/core:core
//tensorflow_model_optimization/python/core:version
//tensorflow_model_optimization/python/core/api:api
...
Loading: 35 packages loaded
```

#. display more logs when building targets: ``bazel build --verbose_failures //tensorflow:tensorflow_cc``

#. how to run test: ``bazel test //tensorflow_model_optimization/python/core/quantization/keras:quantize_functional_test`` (inherit all build opts)