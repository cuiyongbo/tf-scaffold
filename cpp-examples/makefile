
# how to build tensorflow c lib: https://www.tensorflow.org/install/lang_c
cur_dir=${shell pwd}
TF_HEADER_DIR = ${cur_dir}/libtensorflow/include
TF_LIBRARY_DIR = ${cur_dir}/libtensorflow/lib

# how to print a variable in makefile: https://stackoverflow.com/questions/16467718/how-to-print-out-a-variable-in-makefile
$(info TF_HEADER_DIR: ${TF_HEADER_DIR})
$(info TF_LIBRARY_DIR: ${TF_LIBRARY_DIR})

# how to change environment var: https://stackoverflow.com/questions/8941110/how-i-could-add-dir-to-path-in-makefile
export LIBRARY_PATH = ${LIBRARY_PATH}:${TF_LIBRARY_DIR}
export DYLD_LIBRARY_PATH := ${DYLD_LIBRARY_PATH}:${TF_LIBRARY_DIR}

#export LIBRARY_PATH=$LIBRARY_PATH:
#export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:

# Undefined symbols for architecture arm64: "_TF_Version", referenced from: ** ld: symbol(s) not found for architecture arm64
# add `-ltensorflow` when building target

CXX = g++
CXXFLAGS += -I${TF_HEADER_DIR} -L${TF_LIBRARY_DIR}

# DO NOT link both of them
#CXXFLAGS += -ltensorflow # link c library
CXXFLAGS += -ltensorflow_cc # link c++ library

# Undefined symbols for architecture arm64: "tensorflow::NewSession
CXXFLAGS += -ltensorflow_framework

# Undefined symbols for architecture arm64: "tsl::io::internal::JoinPathImpl
CXXFLAGS += -std=c++1z # use c++17 standard


all: hello_tf label_image

# makefile:23: *** missing separator.  Stop.
# indent using tab, and convert space to tab
hello_tf: hello_tf.cpp
	${CXX} ${CXXFLAGS} $? -o $@
	#./hello_tf

label_image: label_image.cpp
	${CXX} ${CXXFLAGS} $? -o $@
	./label_image --image=./data/grace_hopper.jpg --graph=./data/inception_v3_2016_08_28_frozen.pb --labels=./data/imagenet_slim_labels.txt

$(info current directiory: ${shell ls})

clean:
	rm -rf hello_tf label_image *.o *.out *.dSYM
