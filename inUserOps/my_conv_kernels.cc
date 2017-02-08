#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/strings/str_util.h"

#include <cmath>


using namespace tensorflow;


int factorial(int x, int result = 1) {
  if (x == 1) return result; else return factorial(x - 1, x * result);
}

class myConvOp : public OpKernel {
    public:
    explicit myConvOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {

        const Tensor & x_tensor = ctx->input(0);
        auto x = x_tensor.flat<float>();

        const Tensor & w_out_tensor = ctx->input(1);
        auto w_out = w_out_tensor.flat<float>();

        const Tensor & w_in_tensor = ctx->input(2);
        auto w_in = w_in_tensor.flat<float>();

        const Tensor & b_tensor = ctx->input(3);
        auto b = b_tensor.flat<float>();

        const Tensor & num_inputs = ctx->input(4);
        unsigned d = (int)num_inputs.scalar<float>()();

        const Tensor & num_hidden = ctx->input(5);
        unsigned h = (int)num_hidden.scalar<float>()();

        float w_ok_prod = 1;

        float w_ik_prod[d];
        std::fill(w_ik_prod, w_ik_prod+d,1); 

        float b_k__w_ik_sum[d];
        std::fill(b_k__w_ik_sum, b_k__w_ik_sum+d,0); 

        float first_part = 0;
        for(int k=0; k<h; ++k) {

            float w_ok = w_out(k);
            w_ok_prod *= w_ok;

            float b_k = b(k);
            float relu = b_k;
            for(int i=0; i<d; ++i) {

                float w_ik = w_in(k*d+i);
                w_ik_prod[i] *= w_ik;
                b_k__w_ik_sum[i] += b_k/w_ik;

                float x_i = x(i);
                relu += w_ik+x_i;
            }
            if(relu > 0) {
                first_part += w_ok*relu;
            }
        }

        // float second_part = 0;
        // for(int i=0; i<d; ++i) {

        //     float to_raise = x(i)+b_k__w_ik_sum[i];
        //     float raised = 0;
        //     for(int l=0; l<2*h+1; ++l) {
        //         raised *= to_raise/((float)l+1);
        //     }

        //     second_part += w_ok_prod*w_ik_prod[i]*raised;
        // }

        float second_part = 0;
        for(int i=0; i<d; ++i) {
            second_part += w_ik_prod[i]*pow(x(i)+b_k__w_ik_sum[i],2*h+1);
        }
        second_part *= w_ok_prod*factorial(2*h+1);


        Tensor ans_tensor(DT_FLOAT,TensorShape({}));
        auto ans = ans_tensor.flat<float>();
        ans(0) = first_part+second_part;


        ctx->set_output(0,ans_tensor);

    }
};
REGISTER_KERNEL_BUILDER(Name("MyConv").Device(DEVICE_CPU), myConvOp);












