#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

// cd ~/tensorflow/tensorflow/core/user_ops
// bazel build -c opt //tensorflow/core/user_ops:asa.so

using namespace tensorflow;


REGISTER_OP("PointGenerator")
    .Input("params: float")
    .Input("param_temps: float")
    .Input("bounds: float")
    .Output("potential_new_point: float")
    .Doc(R"doc(
Generates a new point that might be accepted.

bounds: bounds on the parameters
params: current parameter values
param_temps: current parameter temperatures
potential_new_point: new point that ASA could go to
        )doc");

/* ------------------------------------------------------------------------------ */

REGISTER_OP("AcceptTest")
    .Input("accept_temp: float")
    .Input("new_cost: float")
    .Input("current_cost: float")
    .Output("accepted: bool")
    .Output("just_a_test: float")
    .Doc(R"doc(
Accepts or rejects the point given.

accept_temp: temperature for accepting
current_cost: the current cost for potential_new_point
accepted: true if this point was accepted
        )doc");

/* ------------------------------------------------------------------------------ */

REGISTER_OP("TempAnneal")
    .Input("c: float")
    .Input("q: float")
    .Input("param_temps_initial: float")
    .Input("param_temps_anneal_time: float")
    .Input("accept_temp_initial: float")
    .Input("accept_temp_anneal_time: float")
    .Output("new_param_temps: float")  
    .Output("new_param_temps_anneal_time: float")  
    .Output("new_accept_temp: float")  
    .Output("new_accept_temp_anneal_time: float")  
    .Doc(R"doc(
Updates the temperature annealing parameters

c: a constant
q: quenching factor
param_temps_initial: the initial parameter temperatures
param_temps_anneal_time: original val
accept_temp_initial: the initial acceptance temperature
accept_temp_anneal_time: original val
new_param_temps: the updated parameter
new_param_temps_anneal_time: the updated parameter
new_accept_temp: the updated parameter
new_accept_temp_anneal_time: the updated parameter
        )doc");

/* ------------------------------------------------------------------------------ */

REGISTER_OP("ReAnneal")
    .Input("c: float")
    .Input("best_cost: float")
    .Input("current_cost: float")
    .Input("param_temps_initial: float")
    .Input("param_temps: float")
    .Input("gradients: float")
    .Output("new_param_temps: float")
    .Output("new_param_temps_anneal_time: float")
    .Output("new_accept_temp_initial: float")
    .Output("new_accept_temp: float")
    .Output("new_accept_temp_anneal_time: float")

    .Doc(R"doc(
Re-anneals the annealing parameters

c: a constant
best_cost: cost for best_params
current_cost: the current cost
param_temps_initial: the initial parameter temperatures
param_temps: current parameter temperatures
gradients: gradients
new_param_temps: updated parameter
new_param_temps_anneal_time: updated parameter
new_accept_temp_initial: updated parameter
new_accept_temp: updated parameter
new_accept_temp_anneal_time: updated parameter
        )doc");

/* ------------------------------------------------------------------------------ */

REGISTER_OP("MyFunction")
    .Input("in: float")
    .Output("out: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
The function we are optimizing.

in: Input
out: Output
        )doc");
