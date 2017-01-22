# this file contains functions that will be used to 
# generate points that the neural net will use to initialize
# its annealing function to that of ASA

import operator
import itertools
import random
import time
import numpy as np
import tensorflow as tf
from myASA import *



def generateData():
    # the function that lets us search the space is:
    # sgn(u_i-0.5)*T_i(k_i)*((1+1/T_i(k_i))^abs(2*u_i-1)-1)
    def theFunc(u,t):
        return abs(u-0.5)*t*((1.0+1.0/t)**abs(2.0*u-1.0)-1.0)
    
    # randomly pick a ton of u,t to plug into theFunc
    # u is in [0,1], t is a value above zero
    # this will make a billion training examples
    numU = 1000
    numT = 1000

    allU = [random.random() for u in range(numU)]
    allT = [abs(t) for t in np.random.normal(0,10,numT)]

    # will write all the data to a text file
    string = ''
    with open('workfile_small.txt', 'w') as f:
        for u,t in itertools.product(allU,allT):
            val = theFunc(u,t)
            string += str(t)+','+str(u)+':'+str(val)+'\n'
        f.write(string)

class model():
    def __init__(self):
        self.extractAnnealingFunctionData('workfile_small.txt')
        print('Done extracting data')
        self.testTrainSplit(0.8)
        self.makeGraphToTrainAnnealingFunction()

    def _makeGraphToTrainAnnealingFunction2(self):
        inputSize = 2
        outputSize = 1
        layerSize = 1000
        numLayers = 1
        keep_prob = 1.0

        self.x = tf.placeholder(tf.float32, shape=[None, inputSize])
        self._y = tf.placeholder(tf.float32, shape=[None, outputSize])
        weight = tf.Variable(tf.random_normal([inputSize, layerSize], stddev=0.35),name='weight0')
        bias = tf.Variable(tf.zeros([layerSize]), name='bias0')
        lastLayer = tf.sigmoid(tf.matmul(self.x,weight)+bias)
        if(numLayers > 2):
            for l in range(numLayers-2):
                # keep_prob = tf.constant(keep_prob)
                # lastLayer = tf.nn.dropout(lastLayer,keep_prob)
                weight = tf.Variable(tf.random_normal([layerSize, layerSize], stddev=0.35),name='weight'+str(l+1))
                bias = tf.Variable(tf.zeros([layerSize]), name='bias'+str(l+1))
                lastLayer = tf.sigmoid(tf.matmul(drop,weight)+bias)
        
        # keep_prob = tf.constant(keep_prob)
        # lastLayer = tf.nn.dropout(lastLayer,keep_prob)

        weight = tf.Variable(tf.random_normal([layerSize, outputSize], stddev=0.35),name='weight'+str(numLayers-1))
        bias = tf.Variable(tf.zeros([outputSize]), name='bias'+str(numLayers-1))
        self.y = tf.sigmoid(tf.matmul(lastLayer,weight)+bias)


    def _makeGraphToTrainAnnealingFunction1(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 2])
        self._y = tf.placeholder(tf.float32, shape=[None, 1])
        # self.weight0 = tf.Variable(tf.random_normal([2, 3], stddev=0.35),name='weight0')
        # self.bias0 = tf.Variable(tf.zeros([3]), name='bias0')
        # afterFirstLayer = tf.nn.relu(tf.matmul(self.x,self.weight0)+self.bias0)
        # self.weight1 = tf.Variable(tf.random_normal([3, 1], stddev=0.35),name='weight1')
        # self.bias1 = tf.Variable(tf.zeros([1]), name='bias1')
        
        self.weight0 = tf.Variable([[.106617823,0,.0978335738],[-1.06829703,1.71947777,1.06464219]],name='weight0')
        self.bias0 = tf.Variable([0.99497002,-0.85586923,0.14213698], name='bias0')
        afterFirstLayer = tf.nn.relu(tf.matmul(self.x,self.weight0)+self.bias0)
                
        weight2 = tf.Variable([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],name='weight2')
        bias2 = tf.Variable(tf.zeros([3]), name='bias2')
        layer = tf.matmul(afterFirstLayer,weight2)+bias2

        self.weight1 = tf.Variable([[0.5997116 ],[1.52755487],[-0.64580035]],name='weight1')
        self.bias1 = tf.Variable([-0.05554779], name='bias1')
        self.y = tf.nn.relu(tf.matmul(layer,self.weight1)+self.bias1)


        # weight1 = tf.Variable(tf.random_normal([10, 10], stddev=0.35),name='weight1')
        # bias1 = tf.Variable(tf.zeros([10]), name='bias1')
        # afterSecondLayer = tf.matmul(afterFirstLayer,weight1)+bias1
        # weight2 = tf.Variable(tf.random_normal([10, 10], stddev=0.35),name='weight2')
        # bias2 = tf.Variable(tf.zeros([10]), name='bias2')
        # afterThirdLayer = tf.matmul(afterSecondLayer,weight2)+bias2
        # weight3 = tf.Variable(tf.random_normal([10, 10], stddev=0.35),name='weight3')
        # bias4 = tf.Variable(tf.zeros([10]), name='bias4')
        # afterFourthLayer = tf.matmul(afterThirdLayer,weight3)+bias4
        # weight4 = tf.Variable(tf.random_normal([10, 1], stddev=0.35),name='weight4')
        # bias4 = tf.Variable(tf.zeros([1]), name='bias4')
        # self.y = tf.sigmoid(tf.matmul(afterFourthLayer,weight4)+bias4)

    def makeGraphToTrainAnnealingFunction(self):
        # will have different options
        self._makeGraphToTrainAnnealingFunction1()
        self.loss = tf.reduce_mean(tf.pow(self.y-self._y,2))
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def extractAnnealingFunctionData(self,fileName):
        allData = []
        allLabels = []
        with open(fileName,'r') as f:
            lines = f.readlines()
            for l in lines:
                tu,v = l.split(':')
                t,u = tu.split(',')
                allData.append([t,u])
                allLabels.append(v)

        self._allData = np.array(allData)
        self._allLabels = np.array(allLabels)

        self._allData = np.reshape(self._allData,(self._allData.shape[0],-1))
        self._allLabels = np.reshape(self._allLabels,(self._allLabels.shape[0],-1))

    def testTrainSplit(self,ratio):
        def unison_shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        self._allData,self._allLabels = unison_shuffled_copies(self._allData,self._allLabels)

        numTrain = int(self._allData.shape[0]*ratio)
        self.trainX = self._allData[:numTrain]
        self.trainY = self._allLabels[:numTrain]

        self.testX = self._allData[numTrain:]
        self.testY = self._allLabels[numTrain:]

    def getBatchOfTrainingData(self,batchSize):
        indices = np.random.choice(self.trainX.shape[0],batchSize,replace=False)
        return [self.trainX[indices],self.trainY[indices]]

    def train(self):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(10000):

                batch = self.getBatchOfTrainingData(10000)
                self.opt.run(feed_dict={self.x:batch[0], self._y:batch[1]})

                if(i%150 == 0):
                    start = time.time()
                    trainLoss = 0#self.loss.eval(feed_dict={self.x:batch[0], self._y:batch[1]})
                    testLoss = self.loss.eval(feed_dict={self.x:self.testX, self._y:self.testY})
                    print('\n\n'+str(i)+': '+str(trainLoss)+' '+str(testLoss))
                    # print(self.weight0.eval())
                    # print(self.bias0.eval())
                    # print(self.weight1.eval())
                    # print(self.bias1.eval())

                    vals = sess.run([self.weight0,self.bias0,self.weight1,self.bias1])
                    print(vals[0])
                    print(vals[1])
                    print(vals[2])
                    print(vals[3])

                    end = time.time()
                    print('ELAPSED TIME: '+str(end - start))

                    
                    
# generateData()

m = model()
m.train()














