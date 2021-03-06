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

    def __init__(self,f,sess,dims):
        self._sess = sess
        self._f = f


        self.init_params = tf.placeholder(tf.float32, shape=(dims,))                              # (dims,)
        self.init_param_temps_initial = tf.placeholder(tf.float32, shape=(dims,))                 # (dims,)
        self.init_accept_temp_initial = tf.placeholder(tf.float32, shape=())                      # ()

        self.init_current_cost = self._f(self.init_params)                                        # ()

        self.init_numb_accepted = tf.Variable(0.0,dtype=tf.float32)                                # ()
        self.init_iters = tf.Variable(0.0,dtype=tf.float32)                                       # ()
        self.init_best_params = tf.Variable(self.init_params)                                     # (dims,)
        self.init_best_cost = tf.Variable(self.init_current_cost)                                 # ()
        self.init_param_temps = tf.Variable(self.init_param_temps_initial)                        # (dims,)
        self.init_param_temps_anneal_time = tf.Variable(np.zeros((dims,)),dtype=tf.float32)       # (dims,)         
        self.init_accept_temp = tf.Variable(self.init_accept_temp_initial)                        # () 
        self.init_accept_temp_anneal_time = tf.Variable(0.0,dtype=tf.float32)                     # ()

        initVals = [self.init_params,\
                    self.init_current_cost,\
                    self.init_numb_accepted,\
                    self.init_iters,\
                    self.init_best_params,\
                    self.init_best_cost,\
                    self.init_param_temps_initial,\
                    self.init_param_temps,\
                    self.init_param_temps_anneal_time,\
                    self.init_accept_temp_initial,\
                    self.init_accept_temp,\
                    self.init_accept_temp_anneal_time]

        initShapes = [x.get_shape() for x in initVals]

        self._queue = tf.FIFOQueue(12,[tf.float32 for _ in initVals],shapes=initShapes)
        self._queue_init = self._queue.enqueue(initVals)

        self._params,\
        self._current_cost,\
        self._numb_accepted,\
        self._iters,\
        self._best_params,\
        self._best_cost,\
        self._param_temps_initial,\
        self._param_temps,\
        self._param_temps_anneal_time,\
        self._accept_temp_initial,\
        self._accept_temp,\
        self._accept_temp_anneal_time = self._queue.dequeue()
        
        
        self._param_bounds = tf.placeholder(tf.float32,name='bounds',shape=(2,dims))
        self._dims = dims
        self._c = tf.placeholder(tf.float32,name='c',shape=())
        self._q = tf.placeholder(tf.float32,name='q',shape=())


        self._totalIters = tf.placeholder(tf.float32,name='totalIters',shape=())
        self._acceptUntilReAnneal = tf.placeholder(tf.float32,name='acceptUntilReAnneal',shape=())       
        self._itersUntilTempAnneal = tf.placeholder(tf.float32,name='itersUntilTempAnneal',shape=())     

        self.build_graph()

    def build_graph(self):
        
        self.train_step()
        self._sess.run(tf.global_variables_initializer())

    #########################################################################################################################


    def generatePoint(self):
        new_params = asa_module.point_generator(self._params,self._param_temps,self._param_bounds)
        new_cost = self._f(new_params)
        return new_params,new_cost

    def acceptanceTest(self,new_cost):
        accepted,count = asa_module.accept_test(self._accept_temp,new_cost,self._current_cost)
        return accepted

    def tempAnneal(self):
        new_param_temps,\
        new_param_temps_anneal_time,\
        new_accept_temp,\
        new_accept_temp_anneal_time = asa_module.temp_anneal(self._c,self._q,\
                                                            self._param_temps_initial,self._param_temps_anneal_time,\
                                                            self._accept_temp_initial,self._accept_temp_anneal_time)

        return [new_param_temps,\
                new_param_temps_anneal_time,\
                new_accept_temp,\
                new_accept_temp_anneal_time]

    def reAnneal(self):
        new_param_temps,\
        new_param_temps_anneal_time,\
        new_accept_temp_initial,\
        new_accept_temp,\
        new_accept_temp_anneal_time = asa_module.re_anneal(self._c,self._best_cost,self._current_cost,\
                                                        self._param_temps_initial,self._param_temps,\
                                                        self.getGradient())
        return [new_param_temps,\
                new_param_temps_anneal_time,\
                new_accept_temp_initial,\
                new_accept_temp,\
                new_accept_temp_anneal_time]

    #########################################################################################################################

    def getGradient(self):
        ans = self._f(self._params)
        grad = tf.gradients(ans,self._params)
        return grad

    def tryPoint(self):
        new_params,new_cost = self.generatePoint()
        accepted = self.acceptanceTest(new_cost)
        return new_params,new_cost,accepted

    def tryUpdateBest(self,new_params,new_cost):
        predUpdateBest = tf.less(new_cost,self._best_cost,name='predUpdateBest')
        def updateBest():
            updated_best_cost = new_cost
            updated_best_params = new_params
            return [updated_best_cost,\
                    updated_best_params]

        cond = tf.cond(predUpdateBest,updateBest,lambda:[self._best_cost,\
                                                         self._best_params],name='updateBestParams')
        return cond

    def tryTempAnneal(self,iters):
        predTempAnneal = tf.equal(iters,self._itersUntilTempAnneal,name='predTempAnneal')
        def tempAnneal_():
            new_param_temps,new_param_temps_anneal_time,new_accept_temp,new_accept_temp_anneal_time = self.tempAnneal()
            new_iters = tf.Variable(0.0,name='new_iters')
            # new_iters = iters.assign(0)
            return [new_param_temps,\
                    new_param_temps_anneal_time,\
                    new_accept_temp,\
                    new_accept_temp_anneal_time,\
                    new_iters]

        cond = tf.cond(predTempAnneal,tempAnneal_, lambda: [self._param_temps,\
                                                            self._param_temps_anneal_time,\
                                                            self._accept_temp,\
                                                            self._accept_temp_anneal_time,\
                                                            iters], name='tempAnnealStep')
        return cond

    def tryReAnneal(self,numb_accepted):
        predReAnneal = tf.equal(numb_accepted,self._acceptUntilReAnneal,name='predReAnneal')
        def reAnneal_():
            new_param_temps,new_param_temps_anneal_time,new_accept_temp_initial,new_accept_temp,new_accept_temp_anneal_time = self.reAnneal()
            new_numb_accepted = tf.Variable(0.0,name='new_numb_accepted')
            # new_numb_accepted = numb_accepted.assign(0)
            return [new_param_temps,\
                    new_param_temps_anneal_time,\
                    new_accept_temp_initial,\
                    new_accept_temp,\
                    new_accept_temp_anneal_time,\
                    new_numb_accepted]
        cond = tf.cond(predReAnneal,reAnneal_, lambda: [self._param_temps,\
                                                        self._param_temps_anneal_time,\
                                                        self._accept_temp_initial,\
                                                        self._accept_temp,\
                                                        self._accept_temp_anneal_time,\
                                                        numb_accepted], name='reAnnealStep')

        return cond

    #########################################################################################################################

    def train_step(self):

        new_params,new_cost,accepted = self.tryPoint()
        new_iters = self._iters+1.0

        def weAccepted(new_params_,newCost_):
            numb_accepted = self._numb_accepted+1.0
            updated_best_cost,updated_best_params = self.tryUpdateBest(new_params_,newCost_)
            return [numb_accepted,\
                    updated_best_cost,\
                    updated_best_params]

        numb_accepted,\
        updated_best_cost,\
        updated_best_params = tf.cond(accepted,lambda: weAccepted(new_params,new_cost),lambda:[self._numb_accepted,\
                                                                                            self._best_cost,\
                                                                                            self._best_params], name='checkAccepted')
        new_param_temps,\
        new_param_temps_anneal_time,\
        new_accept_temp,\
        new_accept_temp_anneal_time,\
        new_iters2 = self.tryTempAnneal(new_iters)

        new_param_temps,\
        new_param_temps_anneal_time,\
        new_accept_temp_initial,\
        new_accept_temp,\
        new_accept_temp_anneal_time,\
        new_numb_accepted = self.tryReAnneal(numb_accepted)

        allVars =  [new_params,\
                    new_cost,\
                    new_numb_accepted,\
                    new_iters2,\
                    updated_best_params,\
                    updated_best_cost,\
                    self._param_temps_initial,\
                    new_param_temps,\
                    new_param_temps_anneal_time,\
                    new_accept_temp_initial,\
                    new_accept_temp,\
                    new_accept_temp_anneal_time]

        addBackOn = self._queue.enqueue(allVars)

        return addBackOn


    def myRun(self,feedDict):
        return
        self._sess.run(tf.global_variables_initializer(),feed_dict=feedDict)
        self._sess.run(self._queue_init,feed_dict=feedDict)
        self._sess.run(self.train_step(),feed_dict=feedDict)
        return

        # i = 0
        # sess.run(self._queue_init,feed_dict=feedDict)
        # while(i < 1):
        #     sess.run(self.train_step(),feed_dict={asa._param_bounds: np.array([[-10.0,-10.0,-10.0,-10.0],[10.0,10.0,10.0,10.0]]),\
        #                                         asa._c: 1.0,\
        #                                         asa._q: 1.0,\
        #                                         asa._totalIters: 10.0,\
        #                                         asa._acceptUntilReAnneal: 10.0,\
        #                                         asa._itersUntilTempAnneal: 10.0})
        #     i += 1


f = asa_module.my_function

with tf.Session() as sess:
    asa = adaptiveSimulatedAnnealingModel(f,sess,4)
    
    # tensorboard --logdir=.
    # writer = tf.summary.FileWriter('.', graph=tf.get_default_graph())
    
    feedDict = {
        asa._param_bounds: np.array([[-10.0,-10.0,-10.0,-10.0],[10.0,10.0,10.0,10.0]]),
        asa._c: 1.0,
        asa._q: 1.0,
        asa._totalIters: 10.0,
        asa._acceptUntilReAnneal: 10.0,
        asa._itersUntilTempAnneal: 10.0,

        asa.init_params : np.random.random(4),
        asa.init_param_temps_initial: np.zeros((4,)),
        asa.init_accept_temp_initial: 0.0
    }
    

    asa.myRun(feedDict)









