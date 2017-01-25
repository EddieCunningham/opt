#include "tensorflow/core/framework/op_kernel.h"

#include <cmath>

// g++ -std=c++11 -undefined dynamic_lookup -shared asa_kernels.cc asa_ops.cc -o $TENSORFLOW/core/user_ops/asa.so -fPIC -I $TF_INC -O2

using namespace tensorflow;



class PointGeneratorOp : public OpKernel {
public:
    explicit PointGeneratorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor & params = ctx->input(0);
        // OP_REQUIRES(ctx, TensorShapeUtils::IsVector(params.shape()), errors::InvalidArgument("Must be a vector"));
        auto params_flat = params.flat<float>();

        const Tensor & param_temps = ctx->input(1);
        // OP_REQUIRES(ctx, TensorShapeUtils::IsVector(param_temps.shape()), errors::InvalidArgument("Must be a vector"));
        auto param_temps_flat = param_temps.flat<float>();

        const Tensor & bounds = ctx->input(2);
        // OP_REQUIRES(ctx, TensorShapeUtils::IsVector(bounds.shape()), errors::InvalidArgument("Must be a vector"));
        auto bounds_flat = bounds.matrix<float>();

        unsigned dims = params_flat.size();


        Tensor potential_new_point(DT_FLOAT, TensorShape({dims}));
        auto pnp_buf = potential_new_point.flat<float>();

        for(int i=0; i<dims; ++i) {
            auto lower_bound = bounds_flat(0,i);
            auto upper_bound = bounds_flat(1,i);

            auto current_param = params_flat(i);
            auto current_temp = param_temps_flat(i);

            double new_param;

            for(int j=0; j==0 or (new_param <= lower_bound or new_param >= upper_bound); ++j) {
                double u = rand()/(double)RAND_MAX;
                int sign = 2.0*int(u-0.5 >= 0)-1.0;
                double dParam = sign * current_temp * (pow(1.0 + 1.0/current_temp, std::abs(2.0*u - 1.0)) - 1.0);
                new_param = current_param + dParam*(upper_bound - lower_bound);
            }

            pnp_buf(i) = new_param;
        }

        ctx->set_output(0, potential_new_point);
    }
};

REGISTER_KERNEL_BUILDER(Name("PointGenerator").Device(DEVICE_CPU), PointGeneratorOp);

/* ------------------------------------------------------------------------------ */

class AcceptTestOp : public OpKernel {
private:
    float internalCount = 0;
public:
    explicit AcceptTestOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {

        TTypes<float>::ConstScalar accept_temp = ctx->input(0).scalar<float>();

        TTypes<float>::ConstScalar new_cost = ctx->input(1).scalar<float>();

        TTypes<float>::ConstScalar current_cost = ctx->input(2).scalar<float>();

        float pAccept = 1/(1+exp((new_cost(0)-current_cost(0))/accept_temp(0)));

        Tensor accepted(DT_BOOL,TensorShape({}));
        auto Taccepted = accepted.flat<bool>();

        Taccepted(0) = (pAccept >= rand()/(double)RAND_MAX);

        internalCount += 1;
        Tensor ic(DT_FLOAT,TensorShape({}));
        auto Tic = ic.flat<float>();
        Tic(0) = internalCount;

        ctx->set_output(0, accepted);
        ctx->set_output(1, ic);
    }
};

REGISTER_KERNEL_BUILDER(Name("AcceptTest").Device(DEVICE_CPU), AcceptTestOp);

/* ------------------------------------------------------------------------------ */

class ReAnnealOp : public OpKernel {
public:
    explicit ReAnnealOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {

        const Tensor & c = ctx->input(0);
        auto Tc = c.scalar<float>()();
        
        /* ----------------------------------------------------------------- */

        const Tensor & best_cost = ctx->input(1);
        auto Tbest_cost = best_cost.scalar<float>()();
        
        /* ----------------------------------------------------------------- */

        const Tensor & current_cost = ctx->input(2);
        auto Tcurrent_cost = current_cost.scalar<float>()();
        
        /* ----------------------------------------------------------------- */

        Tensor param_temps_initial = ctx->input(3);
        auto Tparam_temps_initial = param_temps_initial.flat<float>();
        
        /* ----------------------------------------------------------------- */

        Tensor param_temps = ctx->input(4);
        auto Tparam_temps = param_temps.flat<float>();
        
        /* ----------------------------------------------------------------- */

        Tensor param_temps_anneal_time = ctx->input(5);
        auto Tparam_temps_anneal_time = param_temps_anneal_time.flat<float>();

        /* ----------------------------------------------------------------- */

        Tensor accept_temp_initial =  ctx->input(6);
        auto Taccept_temp_initial = accept_temp_initial.flat<float>();
        
        /* ----------------------------------------------------------------- */
        Tensor accept_temp = ctx->input(7); 
        auto Taccept_temp = accept_temp.flat<float>();
        
        /* ----------------------------------------------------------------- */
        Tensor accept_temp_anneal_time = ctx->input(8); 
        auto Taccept_temp_anneal_time = accept_temp_anneal_time.flat<float>();
        
        /* ----------------------------------------------------------------- */

        const Tensor & gradients = ctx->input(9);
        auto Tgradients = gradients.flat<float>();

        /* ----------------------------------------------------------------- */
        
        unsigned dims = Tparam_temps_initial.size();

        float grad_max = -1;
        for(int i=0; i<dims; ++i) {
            if(std::abs(Tgradients(i)) > grad_max) {
                grad_max = std::abs(Tgradients(i));
            }
        }

        for(int i=0; i<dims; ++i) {
            auto grad_i = Tgradients(i);
            if(grad_i != 0) {
                Tparam_temps(i) *= std::abs(grad_max/grad_i);
            }
            Tparam_temps_anneal_time(i) = pow(-1.0/Tc*std::log(Tparam_temps(i)/Tparam_temps_initial(i)),dims);
        }
        Taccept_temp(0) = Tbest_cost;
        Taccept_temp_initial(0) = Tcurrent_cost;
        Taccept_temp_anneal_time(0) = pow(-1.0/Tc*std::log(Taccept_temp(0)/Taccept_temp_initial(0)), dims);
    }
};

REGISTER_KERNEL_BUILDER(Name("ReAnneal").Device(DEVICE_CPU), ReAnnealOp);

/* ------------------------------------------------------------------------------ */

class TempAnnealOp : public OpKernel {
    public:
    explicit TempAnnealOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {


        const Tensor & c = ctx->input(0);
        auto Tc = c.scalar<float>()();

        /* -------------------------------------------------------------------- */

        const Tensor & q = ctx->input(1);
        auto Tq = q.scalar<float>()();

        /* -------------------------------------------------------------------- */

        const Tensor & param_temps_initial = ctx->input(2);
        auto Tparam_temps_initial = param_temps_initial.flat<float>();

        /* -------------------------------------------------------------------- */

        Tensor param_temps = ctx->input(3);
        auto Tparam_temps = param_temps.flat<float>();

        /* -------------------------------------------------------------------- */

        Tensor param_temps_anneal_time = ctx->input(4);
        auto Tparam_temps_anneal_time = param_temps_anneal_time.flat<float>();

        /* -------------------------------------------------------------------- */

        const Tensor & accept_temp_initial = ctx->input(5);
        auto Taccept_temp_initial = accept_temp_initial.scalar<float>()();

        /* -------------------------------------------------------------------- */

        Tensor accept_temp = ctx->input(6);
        auto Taccept_temp = accept_temp.flat<float>();

        /* -------------------------------------------------------------------- */

        Tensor accept_temp_anneal_time = ctx->input(7);
        auto Taccept_temp_anneal_time = accept_temp_anneal_time.flat<float>();

        /* -------------------------------------------------------------------- */


        unsigned dims = Tparam_temps_initial.size();
        for(int i=0; i<dims; ++i) {
            Tparam_temps_anneal_time(i) += 1;
            Tparam_temps(i) = Tparam_temps_initial(i)*exp(-Tc*pow(Tparam_temps_anneal_time(i),Tq/dims));
        }

        Taccept_temp_anneal_time(0) += 1;
        Taccept_temp(0) = Taccept_temp_initial*exp(-Tc*pow(Taccept_temp_anneal_time(0),Tq/dims));
    }
};

REGISTER_KERNEL_BUILDER(Name("TempAnneal").Device(DEVICE_CPU), TempAnnealOp);

/* ------------------------------------------------------------------------------ */


class MyFunctionOp : public OpKernel {
    public:
    explicit MyFunctionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor & x = ctx->input(0);
        auto Tx = x.flat<float>();

        Tensor ans(DT_FLOAT,TensorShape({}));
        auto Tans = ans.flat<float>();
        Tans(0) = 0;

        for(int i=0; i<Tx.size(); ++i) {
            Tans(0) += Tx(i)*Tx(i);
        }

        ctx->set_output(0,ans);

    }
};
REGISTER_KERNEL_BUILDER(Name("MyFunction").Device(DEVICE_CPU), MyFunctionOp);












