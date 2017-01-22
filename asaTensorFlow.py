import operator
import itertools
import random
import time
import numpy as np
import tensorflow as tf


asa_module = tf.load_op_library('/Users/Eddie/.virtualenvs/opt/lib/python2.7/site-packages/tensorflow/core/user_ops/asa.so')
with tf.Session(''):
    
    params = tf.Variable([1,2,3,4],name='params')
    paramBounds = tf.Variable([[-10,-10,-10,-10],[10,10,10,10]],name='paramBounds')
    paramTemps = tf.Variable([1,1,1,1],name='paramTemps')

    print('About to try this thing')
    print(asa_module.point_generator(params,paramTemps,bounds=paramBounds).eval())

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

