#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


// REGISTER_OP("AdaptiveSimulatedAnnealingController")
//     .Attr("T: realnumbertype")
//     .Attr("bounds: T")
//     .Input("accepted: bool")
//     .Input("next_point: T")
//     .Output("params: T")
//     .Output("param_temps: T")
//     .Output("best_params: T")
//     .Doc(R"doc(
// The controller for the ASA technique.

// bounds: a 2xn tensor that has the min and max vals for the parameters
// accepted: whether or not next_point was accepted
// next_point: point returned by acceptTest
// params: current parameters
// param_temps: temperatures for each parameter
//         )doc");

REGISTER_OP("PointGenerator")
    .Input("params: Ref(float)")
    .Input("param_temps: Ref(float)")
    .Input("bounds: Ref(float)")
    .Output("potential_new_point: float")
    .Doc(R"doc(
Generates a new point that might be accepted.

bounds: bounds on the parameters
params: current parameter values
param_temps: current parameter temperatures
potential_new_point: new point that ASA could go to
        )doc");

REGISTER_OP("AcceptTest")
    .Input("potential_new_point: float")
    .Input("accept_temp: float")
    .Input("new_cost: float")
    .Input("current_cost: float")
    .Output("accepted: bool")
    .Output("just_a_test: float")
    .Doc(R"doc(
Accepts or rejects the point given.

potential_new_point: point we are testing
accept_temp: temperature for accepting
current_cost: the current cost for potential_new_point
accepted: true if this point was accepted
        )doc");

REGISTER_OP("ReAnneal")
    .Input("c: float")
    .Input("best_cost: Ref(float)")
    .Input("current_cost: Ref(float)")
    .Input("param_temps_initial: Ref(float)")
    .Input("param_temps: Ref(float)")
    .Input("param_temps_anneal_time: Ref(float)")
    .Input("accept_temp_initial: Ref(float)")
    .Input("accept_temp: Ref(float)")
    .Input("accept_temp_anneal_time: Ref(float)")
    .Input("gradients: Ref(float)")
    .Doc(R"doc(
Re-anneals the annealing parameters

c: a constant
best_cost: cost for best_params
current_cost: the current cost
param_temps_initial: the initial parameter temperatures
param_temps: current parameter temperatures
param_temps_anneal_time: current parameter annealing time
accept_temp_initial: the initial acceptance temperature
accept_temp: current acceptance temperature
accept_temp_anneal_time: current acceptance annealing time
gradients: gradients
        )doc");

// REGISTER_OP("TempAnneal")
//     .Attr("T: realnumbertype") 
//     .Attr("c: T")
//     .Attr("q: T")
//     .Input("param_temps_initial: Ref(T)")
//     .Input("param_temps: Ref(T)")
//     .Input("param_temps_anneal_time: Ref(T)")
//     .Input("accept_temp_initial: Ref(T)")
//     .Input("accept_temp: Ref(T)")
//     .Input("accept_temp_anneal_time: Ref(T)")    
//     .Doc(R"doc(
// Updates the temperature annealing parameters

// c: a constant
// q: quenching factor
// param_temps_initial: the initial parameter temperatures
// param_temps: current parameter temperatures
// param_temps_anneal_time: current parameter annealing time
// accept_temps_initial: the initial acceptance temperature
// accept_temps: current acceptance temperature
// accept_temps_anneal_time: current acceptance annealing time
//         )doc");

REGISTER_OP("MyFunction")
    .Input("in: Ref(float)")
    .Output("out: float")
    .Doc(R"doc(
The function we are optimizing.

in: Input
out: Output
        )doc");

Status MyFunctionGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {

	auto dx = Const(scope, 1.0);
	grad_outputs->push_back(dx);
	return scope.status();
}
REGISTER_GRADIENT_OP("MyFunction", MyFunctionGrad);




