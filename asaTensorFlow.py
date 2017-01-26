import operator
import itertools
import random
import time
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

@ops.RegisterGradient("MyFunction")
def _my_function_grad(op, grad):
    return [2*tf.multiply(op.inputs[0],grad)]


asa_module = tf.load_op_library('../tensorflow/bazel-bin/tensorflow/core/user_ops/asa.so')

class adaptiveSimulatedAnnealingModel(object):

    # def __init__(self,params_0,bounds,f,sess):
    #     self._sess = sess

    #     # q = tf.FIFOQueue(1,'float')
    #     # init = q.enqueue(params_0)

    #     # self._params = q.dequeue()
    #     self._params = tf.Variable(params_0,name='params',dtype=tf.float32)
        
    #     self._param_bounds = tf.Variable(bounds,name='bounds',dtype=tf.float32)
    #     self._dims = tf.TensorShape(self._param_bounds.get_shape()[1])
    #     self._f = f
    #     self._c = tf.placeholder(tf.float32,name='c')
    #     self._q = tf.placeholder(tf.float32,name='q')
    #     self._current_cost = tf.Variable(self._f(self._params.initialized_value()),name='current_cost',dtype=tf.float32)


    #     self._totalIters = tf.placeholder(tf.float32,name='totalIters')

    #     self._numbAccepted = tf.Variable(0,name='numbAccepted',dtype=tf.float32)
    #     self._acceptUntilReAnneal = tf.placeholder(tf.float32,name='acceptUntilReAnneal')       

    #     self._iters = tf.Variable(0,name='iters',dtype=tf.float32)
    #     self._itersUntilTempAnneal = tf.placeholder(tf.float32,name='itersUntilTempAnneal')     

    #     self.build_graph()

    # def build_graph(self):
        
    #     self._best_params = tf.Variable(self._params.initialized_value())
    #     self._best_cost = tf.Variable(self._current_cost.initialized_value())

    #     self._param_temps_initial = tf.Variable(np.ones(self._dims),name='param_temps_initial',dtype=tf.float32)
    #     self._param_temps = tf.Variable(self._param_temps_initial.initialized_value(),name='param_temps',dtype=tf.float32)
    #     self._param_temps_anneal_time = tf.Variable(np.zeros(self._dims),name='param_temps_anneal_time',dtype=tf.float32)

    #     self._accept_temp_initial = tf.Variable(self._f(self._params.initialized_value()),name='accept_temp_initial',dtype=tf.float32)
    #     self._accept_temp = tf.Variable(self._accept_temp_initial.initialized_value(),name='accept_temp',dtype=tf.float32)
    #     self._accept_temp_anneal_time = tf.Variable(0,name='accept_temp_anneal_time',dtype=tf.float32)

    #     self.train_step()

    #########################################################################################################################

    def __init__(self,f,sess):
        self._sess = sess

        self.init_params = tf.placeholder(tf.float32)
        self.init_current_cost = tf.placeholder(tf.float32)
        self.init_numbAccepted = tf.placeholder(tf.float32)
        self.init_iters = tf.placeholder(tf.float32)
        self.init_best_params = tf.placeholder(tf.float32)
        self.init_best_cost = tf.placeholder(tf.float32)
        self.init_param_temps_initial = tf.placeholder(tf.float32)
        self.init_param_temps = tf.placeholder(tf.float32)
        self.init_param_temps_anneal_time = tf.placeholder(tf.float32)
        self.init_accept_temp_initial = tf.placeholder(tf.float32)
        self.init_accept_temp = tf.placeholder(tf.float32)
        self.init_accept_temp_anneal_time = tf.placeholder(tf.float32)

        initVals = [self.init_params,self.init_current_cost,self.init_numbAccepted,self.init_iters,self.init_best_params,self.init_best_cost,self.init_param_temps_initial,self.init_param_temps,self.init_param_temps_anneal_time,self.init_accept_temp_initial,self.init_accept_temp,self.init_accept_temp_anneal_time]

        self._queue = tf.FIFOQueue(12,'float')
        self._queue_init = self._queue.enqueue(initVals)

        self._params,self._current_cost,self._numbAccepted,self._iters,self._best_params,self._best_cost,self._param_temps_initial,self._param_temps,self._param_temps_anneal_time,self._accept_temp_initial,self._accept_temp,self._accept_temp_anneal_time = q.dequeue()
        
        
        self._param_bounds = tf.placeholder(tf.float32,name='bounds')
        self._dims = tf.TensorShape(self._param_bounds.get_shape()[1])
        self._f = f
        self._c = tf.placeholder(tf.float32,name='c')
        self._q = tf.placeholder(tf.float32,name='q')


        self._totalIters = tf.placeholder(tf.float32,name='totalIters')
        self._acceptUntilReAnneal = tf.placeholder(tf.float32,name='acceptUntilReAnneal')       
        self._itersUntilTempAnneal = tf.placeholder(tf.float32,name='itersUntilTempAnneal')     

        self.build_graph()

    def build_graph(self):
        
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
            return [self._best_cost,self._best_params]
        return tf.cond(predUpdateBest,updateBest,lambda:[self._best_cost,self._best_params],name='updateBestParams')

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
        self._iters.assign_add(1)

        def weAccepted(newPoint_,newCost_):
            self._numbAccepted.assign_add(1)
            tub = self.tryUpdateBest(newPoint_,newCost_)
            ans = [self._numbAccepted]
            ans.extend(tub)
            return ans

        ca = tf.cond(accepted,lambda: weAccepted(newPoint,newCost),lambda:[self._numbAccepted,self._best_cost,self._best_params], name='checkAccepted')
        ta = self.tryTempAnneal()
        ra = self.tryReAnneal()

        allVars = [self._params,self._current_cost,self._numbAccepted,self._iters,self._best_params,self._best_cost,self._param_temps_initial,self._param_temps,self._param_temps_anneal_time,self._accept_temp_initial,self._accept_temp,self._accept_temp_anneal_time]
        addBackOn = self._queue.enqueue_many(allVars)

        return addBackOn


    def myRun(self,feedDict):
        i = 0
        while(i < 10):
            print(sess.run(self.train_step(),feed_dict=feedDict))
            i += 1

# params_0 = np.random.random(4)
# bounds = np.array([[-10.0,-10.0,-10.0,-10.0],[10.0,10.0,10.0,10.0]])
# f = asa_module.my_function
# c = 1.0
# q = 1.0

# with tf.Session() as sess:
#     asa = adaptiveSimulatedAnnealingModel(params_0,bounds,f,sess)
#     sess.run(tf.global_variables_initializer())
    
#     # writer = tf.summary.FileWriter('.', graph=tf.get_default_graph())
    

#     feedDict = {
#         asa._c: 1.0,
#         asa._q: 1.0,
#         asa._totalIters: 10.0,
#         asa._acceptUntilReAnneal: 10.0,
#         asa._itersUntilTempAnneal: 10.0
#     }
#     asa.myRun(feedDict)



f = asa_module.my_function


with tf.Session() as sess:
    asa = adaptiveSimulatedAnnealingModel(f,sess)
    sess.run(tf.global_variables_initializer())
    
    # writer = tf.summary.FileWriter('.', graph=tf.get_default_graph())
    

    feedDict = {
        asa._param_bounds: np.array([[-10.0,-10.0,-10.0,-10.0],[10.0,10.0,10.0,10.0]])
        asa._c: 1.0,
        asa._q: 1.0,
        asa._totalIters: 10.0,
        asa._acceptUntilReAnneal: 10.0,
        asa._itersUntilTempAnneal: 10.0,

        asa.init_params : ,
        asa.init_current_cost : ,
        asa.init_numbAccepted : ,
        asa.init_iters : ,
        asa.init_best_params : ,
        asa.init_best_cost : ,
        asa.init_param_temps_initial : ,
        asa.init_param_temps : ,
        asa.init_param_temps_anneal_time : ,
        asa.init_accept_temp_initial : ,
        asa.init_accept_temp : ,
        asa.init_accept_temp_anneal_time : 
    }
    asa.myRun(feedDict)








