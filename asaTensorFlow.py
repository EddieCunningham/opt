import operator
import itertools
import random
import time
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

@ops.RegisterGradient("MyFunction")
def _my_function_grad(op, grad):
    # print('op is: '+str(op))
    # print('grad is: '+str(grad))
    # print('ans is: '+str(2*tf.multiply(op.inputs[0],grad)))
    return [2*tf.multiply(op.inputs[0],grad)]


asa_module = tf.load_op_library('../tensorflow/bazel-bin/tensorflow/core/user_ops/asa.so')

# params = tf.Variable([1,2,3,4],name='params',dtype=tf.float32)
# param_bounds = tf.Variable([[-10,-10,-10,-10],[10,10,10,10]],name='param_bounds',dtype=tf.float32)
# param_temps = tf.Variable([1,1,1,1],name='param_temps',dtype=tf.float32)

# potential_new_point = asa_module.point_generator(params,param_temps,param_bounds)

# accept_temp = tf.Variable([2.0],dtype=tf.float32)

# new_cost = tf.Variable([5.0],dtype=tf.float32)

# current_cost = tf.Variable([4.0],dtype=tf.float32)

# accepted,count = asa_module.accept_test(accept_temp,new_cost,current_cost)

# c = tf.Variable([1.0],name='c',dtype=tf.float32)
# q = tf.Variable([1.0],name='q',dtype=tf.float32)
# best_cost = tf.Variable([1.0],name='c',dtype=tf.float32)
# # current_cost
# param_temps_initial = tf.Variable([1,1,1,1],name='param_temps_initial',dtype=tf.float32)
# # param_temps
# param_temps_anneal_time = tf.Variable([1,1,1,1],name='param_temps_anneal_time',dtype=tf.float32)
# accept_temp_initial = tf.Variable([1.0],name='accept_temp_initial',dtype=tf.float32)
# accept_temp = tf.Variable([5.0],name='accept_temp',dtype=tf.float32)
# accept_temp_anneal_time = tf.Variable([1.0],name='accept_temp_anneal_time',dtype=tf.float32)
# grad = tf.Variable([1,1,1,1],name='param_temps_initial',dtype=tf.float32)


# reanneal = asa_module.re_anneal(c,best_cost,current_cost,\
#     param_temps_initial,param_temps,param_temps_anneal_time,\
#     accept_temp_initial,accept_temp,accept_temp_anneal_time,\
#     grad)

# tempanneal = asa_module.temp_anneal(c,q,\
#     param_temps_initial,param_temps,param_temps_anneal_time,\
#     accept_temp_initial,accept_temp,accept_temp_anneal_time)


# ans = asa_module.my_function(params)

# # grad = tf.gradients(ans,params)

# writer = tf.summary.FileWriter('.', graph=tf.get_default_graph())

# with tf.Session() as sess:

#     sess.run(tf.global_variables_initializer())

#     print('params: '+str(params))
#     print(sess.run(params))

#     print('ans: '+str(ans))
#     print(sess.run(ans))

#     print('reanneal: '+str(reanneal))
#     print(sess.run(reanneal))

#     print('tempanneal: '+str(tempanneal))
#     print(sess.run(tempanneal))

#     print('accept_temp: '+str(accept_temp))
#     print(sess.run(accept_temp))

#     # print('grad: '+str(grad))
#     # print(sess.run(grad))



class adaptiveSimulatedAnnealingModel(object):
    
    def __init__(self,bounds,f,c,q):
        self._param_bounds = tf.Variable(bounds,name='bounds',dtype=tf.float32)
        self._dims = tf.TensorShape(self._param_bounds.get_shape()[1])
        self._f = f
        self._c = tf.Variable(c,name='c',dtype=tf.float32)
        self._q = tf.Variable(q,name='q',dtype=tf.float32)

        self._totalIters = tf.Variable(100,name='totalIters',dtype=tf.int32)

        self._numbAccepted = tf.Variable(0,name='numbAccepted',dtype=tf.int32)
        self._acceptUntilReAnneal = tf.Variable(10,name='acceptUntilReAnneal',dtype=tf.int32)

        self._iters = tf.Variable(0,name='iters',dtype=tf.int32)
        self._itersUntilTempAnneal = tf.Variable(10,name='itersUntilTempAnneal',dtype=tf.int32)

        self.build_graph()

    def build_graph(self):
        inBetween = tf.random_uniform(self._dims)+(tf.slice(self._param_bounds,[0,0],[1,-1])+tf.slice(self._param_bounds,[1,0],[1,-1]))/2.0
        self._params = tf.Variable(inBetween,name='params',dtype=tf.float32)
        self._current_cost = tf.Variable(self._f(self._params),name='current_cost',dtype=tf.float32)
        
        self._best_params = tf.Variable(self._params)
        self._best_cost = tf.Variable(self._current_cost)

        self._param_temps_initial = tf.Variable(np.ones(self._dims),name='param_temps_initial',dtype=tf.float32)
        self._param_temps = tf.Variable(self._param_temps_initial,name='param_temps',dtype=tf.float32)
        self._param_temps_anneal_time = tf.Variable(np.zeros(self._dims),name='param_temps_anneal_time',dtype=tf.float32)

        self._accept_temp_initial = tf.Variable(self._f(self._params),name='accept_temp_initial',dtype=tf.float32)
        self._accept_temp = tf.Variable(self._accept_temp_initial,name='accept_temp',dtype=tf.float32)
        self._accept_temp_anneal_time = tf.Variable(0,name='accept_temp_anneal_time',dtype=tf.float32)

        self.initGradient()
        self.train_step()

    #########################################################################################################################


    def generatePoint(self):
        newPoint = asa_module.point_generator(self._params,self._param_temps,self._param_bounds)
        newCost = self._f(newPoint)
        return newPoint,newCost

    def acceptanceTest(self,new_cost):
        accepted,count = asa_module.accept_test(self._accept_temp,new_cost,self._current_cost)
        return accepted

    def tempAnneal(self):
        anneal = asa_module.temp_anneal(self._c,self._q,\
                                        self._param_temps_initial,self._param_temps,self._param_temps_anneal_time,\
                                        self._accept_temp_initial,self._accept_temp,self._accept_temp_anneal_time)
        return anneal

    def reAnneal(self):
        reanneal = asa_module.re_anneal(self._c,self._best_cost,self._current_cost,\
                                        self._param_temps_initial,self._param_temps,self._param_temps_anneal_time,\
                                        self._accept_temp_initial,self._accept_temp,self._accept_temp_anneal_time,\
                                        self.getGradient())
        return reanneal

    #########################################################################################################################

    def getGradient(self):
        ans = self._f(self._params)
        grad = tf.gradients(ans,self._params)
        return grad

    def tryPoint(self):
        newPoint,newCost = self.generatePoint()
        accepted = self.acceptanceTest(newCost)
        return newPoint,newCost,accepted

    def tryUpdateBest(self,newPoint,newCost):
        predUpdateBest = tf.less(newCost,self._best_cost,name='predUpdateBest')
        def updateBest():
            self._best_cost.assign(newCost)
            self._best_params.assign(newPoint)
        return tf.cond(predUpdateBest,updateBest,lambda: tf.no_op(),name='updateBestParams')

    def tryTempAnneal(self):
        predTempAnneal = tf.equal(self._iters,self._itersUntilTempAnneal,name='predTempAnneal')
        def tempAnneal_():
            self._iters.assign(0)
            return self.tempAnneal()
        return tf.cond(predTempAnneal,tempAnneal_, lambda: tf.no_op(), name='tempAnnealStep')

    def tryReAnneal(self):
        predReAnneal = tf.equal(self._numbAccepted,self._acceptUntilReAnneal,name='predReAnneal')
        def reAnneal_():
            self._numbAccepted.assign(0)
            return self.reAnneal()
        return tf.cond(predReAnneal,reAnneal_, lambda: tf.no_op(), name='reAnnealStep')

    #########################################################################################################################

    def train_step(self):

        newPoint,newCost,accepted = self.tryPoint()
        self._iters = tf.add(tf.constant(1),self._iters)

        def weAccepted(newPoint_,newCost_):
            self._numbAccepted = tf.add(tf.constant(1),self._numbAccepted)
            self.tryUpdateBest(newPoint_,newCost_)

        ca = tf.cond(accepted,lambda: weAccepted(newPoint,newCost),lambda: tf.no_op(), name='checkAccepted')
        ta = self.tryTempAnneal()
        ra = self.tryReAnneal()

        return [self._iters,ca,ta,ra]


    def run(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            i = 0
            while(t < self._totalIters):
                sess.run(self.train_step)


bounds = np.array([[-10,-10,-10,-10],[10,10,10,10]])
f = asa_module.my_function
c = 1.0
q = 1.0

asa = adaptiveSimulatedAnnealingModel(bounds,f,c,q)
asa.run()









