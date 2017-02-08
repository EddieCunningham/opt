#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

// cd ~/tensorflow/tensorflow/core/user_ops
// bazel build -c opt //tensorflow/core/user_ops:asa.so

using namespace tensorflow;

REGISTER_OP("ReSinh")
    .Input("in: float")
    .Output("out: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Computed a rectified sinh function

in: Input
out: Output
        )doc");
