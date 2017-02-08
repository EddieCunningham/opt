#include "tensorflow/core/framework/op.h"

// cd ~/tensorflow/tensorflow/core/user_ops
// bazel build -c opt //tensorflow/core/user_ops:asa.so

using namespace tensorflow;


REGISTER_OP("MyConv")
    .Input("x: float")
    .Input("w_out: float")
    .Input("w_in: float")
    .Input("b: float")
    .Input("w_c: float")
    .Output("out: float")
    .Doc(R"doc(
A convolution of a single layer network with relu units.

x: d*1 input vector
w_out: h*1 weight vector
w_in: h*d weight matrix
b: h*1 bias vector
w_c: weight on the convolution output
        )doc");
