import operator
import itertools
import random
import time
import numpy as np
import tensorflow as tf


asa_module = tf.load_op_library('/Users/Eddie/.virtualenvs/opt/lib/python2.7/site-packages/tensorflow/core/user_ops/asa.so')


params = tf.Variable([1,2,3,4],name='params',dtype=tf.float32)
paramBounds = tf.Variable([[-10,-10,-10,-10],[10,10,10,10]],name='paramBounds',dtype=tf.float32)
param_temps = tf.Variable([1,1,1,1],name='param_temps',dtype=tf.float32)

potential_new_point = asa_module.point_generator(params,param_temps,paramBounds)

accept_temp = tf.Variable([2.0],dtype=tf.float32)

new_cost = tf.Variable([5.0],dtype=tf.float32)

current_cost = tf.Variable([4.0],dtype=tf.float32)

accepted,count = asa_module.accept_test(potential_new_point,accept_temp,new_cost,current_cost)


#     c
#     best_cost
#     # current_cost
#     param_temps_initial
#     # param_temps
#     param_temps_anneal_time
#     accept_temp_initial
#     # accept_temp
#     accept_temp_anneal_time
#     gradients
# asa_module.re_anneal(c,)

ans = asa_module.my_function(params)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    print('About to try this thing')
    print(sess.run(ans))




# class adaptiveSimulatedAnnealingModel(object):
    
#   def __init__(self,bounds,f):
#       self._bounds = bounds
#       self._f = f
#       self.build_graph()

#   def build_graph(self):
#       parameters,parameter_temps = asa_controller(bounds=self._bounds)

#       current_cost = objective_function(parameters)

#       acceptance_cost = current_cost

#       potential_new_point = point_generator(parameters,parameter_temps,bounds=self._bounds)

#       accepted,accepted_parameters = acceptance_test(potential_new_point,current_cost,acceptance_cost)

#       reanneal:
#           c,q,
#           best_cost,best_x,
#           tempForParams,tempForParamsInitial,annealTimeForTemp,
#           tempForCost,tempForCostInitial,annealTimeForCost

#       tempAnneal:
#           c,q,
#           tempForParams,tempForParamsInitial,annealTimeForTemp,
#           tempForCost,tempForCostInitial,annealTimeForCost

#   def train_step(self):
#       pass

#   def run(self):
#       pass

