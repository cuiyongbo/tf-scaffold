#!/usr/bin/env bash

# how to build tensorflow c lib: https://www.tensorflow.org/install/lang_c
cur_dir=`pwd`
TF_HEADER_DIR=$cur_dir/libtensorflow/include
TF_LIBRARY_DIR=$cur_dir/libtensorflow/lib
echo TF_HEADER_DIR: $TF_HEADER_DIR, TF_LIBRARY_DIR: $TF_LIBRARY_DIR
export LIBRARY_PATH=$LIBRARY_PATH:$TF_LIBRARY_DIR
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$TF_LIBRARY_DIR

# Undefined symbols for architecture arm64: "_TF_Version", referenced from: ** ld: symbol(s) not found for architecture arm64
# add `-ltensorflow` when building target

compiler=g++
target_name='hello_tf'
echo $compiler -I$TF_HEADER_DIR -L$TF_LIBRARY_DIR hello_tf.cpp -o $target_name -ltensorflow
$compiler -I$TF_HEADER_DIR -L$TF_LIBRARY_DIR hello_tf.cpp -o $target_name -ltensorflow

$cur_dir/$target_name