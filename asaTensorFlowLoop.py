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
        with tf.variable_scope('initializerVars'):
            self._sess = sess
            self._f = f

            self.init_params = tf.placeholder(tf.float32, shape=(dims,))                              # (dims,)
            self.init_param_temps_initial = tf.placeholder(tf.float32, shape=(dims,))                 # (dims,)

            self.init_current_cost = self._f(self.init_params)                                        # ()
            self.init_accept_temp_initial = tf.Variable(self.init_current_cost)                       # ()

            self.init_numb_accepted = tf.Variable(0.0,dtype=tf.float32)                               # ()
            self.init_iters = tf.Variable(0.0,dtype=tf.float32)                                       # ()
            self.init_best_params = tf.Variable(self.init_params)                                     # (dims,)
            self.init_best_cost = tf.Variable(self.init_current_cost)                                 # ()
            self.init_param_temps = tf.Variable(self.init_param_temps_initial)                        # (dims,)
            self.init_param_temps_anneal_time = tf.Variable(np.zeros((dims,)),dtype=tf.float32)       # (dims,)         
            self.init_accept_temp = tf.Variable(self.init_accept_temp_initial.initialized_value())    # () 
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

        with tf.variable_scope('queueVars'):
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
            
        with tf.variable_scope('placeholderVars'):
        
            self._param_bounds = tf.placeholder(tf.float32,name='bounds',shape=(2,dims))
            self._dims = dims
            self._c = tf.placeholder(tf.float32,name='c',shape=())
            self._q = tf.placeholder(tf.float32,name='q',shape=())


            self._totalIters = tf.placeholder(tf.float32,name='totalIters',shape=())
            self._acceptUntilReAnneal = tf.placeholder(tf.float32,name='acceptUntilReAnneal',shape=())       
            self._itersUntilTempAnneal = tf.placeholder(tf.float32,name='itersUntilTempAnneal',shape=())    

            self._zeroVar = tf.Variable(0.0,name='zero') 

        self.build_graph()

    def build_graph(self):
        
        self.train_step()

    #########################################################################################################################


    def generatePoint(self):
        with tf.variable_scope('generatePoint'):
            new_params = asa_module.point_generator(self._params,self._param_temps,self._param_bounds)
            new_cost = self._f(new_params)
            return new_params,new_cost

    def acceptanceTest(self,new_cost):
        with tf.variable_scope('acceptanceTest'):
            accepted,count = asa_module.accept_test(self._accept_temp,new_cost,self._current_cost)
            return accepted

    def tempAnneal(self):
        with tf.variable_scope('tempAnneal'):
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
        with tf.variable_scope('reAnneal'):
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


    def getGradient(self):
        with tf.variable_scope('getGradient'):
            ans = self._f(self._params)
            grad = tf.gradients(ans,self._params)
            return grad

    #########################################################################################################################

    def tryUpdateBest(self,new_params,new_cost):
        with tf.variable_scope('tryToUpdateBest'):
            predUpdateBest = tf.less(new_cost,self._best_cost,name='predUpdateBest')

            cond = tf.cond(predUpdateBest,lambda:[new_cost,\
                                                  new_params],\
                                          lambda:[self._best_cost,\
                                                  self._best_params],name='updateBestParams')
            return cond

    def tryMoving(self):
        with tf.variable_scope('tryMoving'):
            new_params_,new_cost_ = self.generatePoint()
            updated_best_cost,updated_best_params = self.tryUpdateBest(new_params_,new_cost_)

            iters_plus_one = self._iters+1.0

            accepted = self.acceptanceTest(new_cost_)
            numb_accepted_plus_one = self._numb_accepted+1.0

            new_params,\
            new_cost,\
            new_iters,\
            new_numb_accepted,\
            new_best_cost,\
            new_best_params = tf.cond(accepted, lambda:[new_params_,\
                                                        new_cost_,\
                                                        iters_plus_one,\
                                                        numb_accepted_plus_one,\
                                                        updated_best_cost,\
                                                        updated_best_params],\
                                                lambda:[self._params,\
                                                        self._current_cost,\
                                                        iters_plus_one,\
                                                        self._numb_accepted,\
                                                        self._best_cost,\
                                                        self._best_params], name='checkAccepted')

            return [new_params,new_cost,new_iters,new_numb_accepted,new_best_cost,new_best_params]

    def tryTempAnneal(self,iters):
        with tf.variable_scope('tryTempAnneal'):
            predTempAnneal = tf.equal(tf.mod(iters,self._itersUntilTempAnneal),self._zeroVar,name='predTempAnneal')

            new_param_temps,new_param_temps_anneal_time,new_accept_temp,new_accept_temp_anneal_time = self.tempAnneal()

            cond = tf.cond(predTempAnneal,lambda:[new_param_temps,\
                                                  new_param_temps_anneal_time,\
                                                  new_accept_temp,\
                                                  new_accept_temp_anneal_time],\
                                          lambda:[self._param_temps,\
                                                  self._param_temps_anneal_time,\
                                                  self._accept_temp,\
                                                  self._accept_temp_anneal_time], name='tempAnnealStep')
            return cond

    def tryReAnneal(self,numb_accepted):
        with tf.variable_scope('tryReAnneal'):
            predReAnneal = tf.equal(tf.mod(numb_accepted,self._acceptUntilReAnneal),self._zeroVar,name='predReAnneal')

            new_param_temps,new_param_temps_anneal_time,new_accept_temp_initial,new_accept_temp,new_accept_temp_anneal_time = self.reAnneal()

            cond = tf.cond(predReAnneal,lambda:[new_param_temps,\
                                                new_param_temps_anneal_time,\
                                                new_accept_temp_initial,\
                                                new_accept_temp,\
                                                new_accept_temp_anneal_time],\
                                        lambda:[self._param_temps,\
                                                self._param_temps_anneal_time,\
                                                self._accept_temp_initial,\
                                                self._accept_temp,\
                                                self._accept_temp_anneal_time], name='reAnnealStep')

            return cond



    #########################################################################################################################

    def train_step(self):

        new_params,\
        new_cost,\
        new_iters,\
        new_numb_accepted,\
        new_best_cost,\
        new_best_params = self.tryMoving()

        #______________________________________________________________________________#

        new_param_temps,\
        new_param_temps_anneal_time,\
        new_accept_temp,\
        new_accept_temp_anneal_time = self.tryTempAnneal(new_iters)

        #______________________________________________________________________________#

        # new_param_temps,\
        # new_param_temps_anneal_time,\
        # new_accept_temp_initial,\
        # new_accept_temp,\
        # new_accept_temp_anneal_time = self.tryReAnneal(new_numb_accepted)

        new_accept_temp_initial = self._accept_temp_initial
        #______________________________________________________________________________#


        allVars =  [new_params,\
                    new_cost,\
                    new_numb_accepted,\
                    new_iters,\
                    new_best_params,\
                    new_best_cost,\
                    self._param_temps_initial,\
                    new_param_temps,\
                    new_param_temps_anneal_time,\
                    new_accept_temp_initial,\
                    new_accept_temp,\
                    new_accept_temp_anneal_time]

        addBackOn = self._queue.enqueue(allVars)

        return addBackOn


    def myRun(self,feedDict):
        self._sess.run(tf.global_variables_initializer(),feed_dict=feedDict)
        self._sess.run(self._queue_init,feed_dict=feedDict)

        for i in range(5):
            self._sess.run(self.train_step(),feed_dict=feedDict)

        self.seeResult(str(self._sess.run([self._params,\
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
                      self._accept_temp_anneal_time])))

    def seeResult(self,outString):
        parsedByArray = outString[1:-1].split('array(')
        moreParts = []
        for p in parsedByArray:
            moreParts.extend(p.split(', dtype=float32)'))

        arrayParts = [moreParts[i] for i in [1,3,5,7,9]]

        scalarParts = [s for s in [moreParts[i].strip(',').strip(' ').strip('\n') for i in [2,4,6,8,10]] if len(s) > 0]
        allScalarParts = []
        for s in scalarParts:
            allScalarParts.extend([_s.strip(' ').strip(',') for _s in s.split(',')])
        allScalarParts = [a for a in allScalarParts if len(a) > 0]

        print('\nparams: '+str(arrayParts[0]))
        print('current_cost: '+str(allScalarParts[0]))
        print('numb_accepted: '+str(allScalarParts[1]))
        print('iters: '+str(allScalarParts[2]))
        print('best_params: '+str(arrayParts[1]))
        print('best_cost: '+str(allScalarParts[3]))
        print('param_temps_initial: '+str(arrayParts[2]))
        print('param_temps: '+str(arrayParts[3]))
        print('param_temps_anneal_time: '+str(arrayParts[4]))
        print('accept_temp_initial: '+str(allScalarParts[4]))
        print('accept_temp: '+str(allScalarParts[5]))
        print('accept_temp_anneal_time: '+str(allScalarParts[6]))
        print('\n')

        




f = asa_module.my_function

with tf.Session() as sess:
    asa = adaptiveSimulatedAnnealingModel(f,sess,4)
    
    # tensorboard --logdir=.
    # writer = tf.summary.FileWriter('.', graph=sess.graph)
    
    feedDict = {
        asa._param_bounds: np.array([[-10.0,-10.0,-10.0,-10.0],[10.0,10.0,10.0,10.0]]),
        asa._c: 10.0,
        asa._q: 1.0,
        asa._totalIters: 100.0,
        asa._acceptUntilReAnneal: 10.0,
        asa._itersUntilTempAnneal: 10.0,

        asa.init_params : np.random.normal(-3,1,4),
        asa.init_param_temps_initial: np.ones((4,))
    }

    asa.myRun(feedDict)









