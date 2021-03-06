import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):
    """The gradients for `zero_out`.

    Args:
    op: The `zero_out` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

    Returns:
    Gradients with respect to the input of `zero_out`.
    """
    to_zero = op.inputs[0]
    shape = array_ops.shape(to_zero)
    index = array_ops.zeros_like(shape)
    first_grad = array_ops.reshape(grad, [-1])[0]
    to_zero_grad = sparse_ops.sparse_to_dense(index, shape, first_grad, 0)
    return [to_zero_grad]  # List of one Tensor, since we have one input





zero_out_module = tf.load_op_library('../tensorflow/bazel-bin/tensorflow/core/user_ops/zero_out.so')
a = tf.Variable([[1, 2], [3, 4]])
b = zero_out_module.zero_out(a)
c = tf.gradients(b,a)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))

