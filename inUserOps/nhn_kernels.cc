#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/strings/str_util.h"

#include <cmath>


using namespace tensorflow;


// This op computes the convolution of multiple nodes.
// What needs to be passed:
//  - input tensor
//  - 

class ReSinhOp : public OpKernel {
    public:
    explicit ReSinhOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor & x = ctx->input(0);
        auto Tx = x.flat<float>();

        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,x.shape(),&output_tensor));
        auto output = output_tensor->flat<int32>();

        const int N = Tx.size();
        for (int i=0; i<N; i++) {
            auto val = Tx(i);
            if(val > 0) {
                output(i) = sinh(val);
            }
            else {
                output(i) = 0;
            }
        }
    }
};
REGISTER_KERNEL_BUILDER(Name("ReSinh").Device(DEVICE_CPU), ReSinhOp);












