load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
)

# run cmd to build the executable: ``bazel build //main:hello-world``
# ``//main `` is the location of target relative to the root directory
# ``hello-world`` is the target name in the BUILD file