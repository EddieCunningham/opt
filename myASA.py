import operator
import sys
import itertools
import scipy.optimize as opt
import random
import scipy.special
import numpy as np

# making this file because ASA is impossible to use efficiently/easily through python
# will use fuzzy ASA
class ASA():
    def __init__(self,f,xMin,xMax,c,q,numbReAnneal,numbTempAnneal,numbIterations):
        self.f = f
        self.xMin = xMin
        self.xMax = xMax

        self.x = 5*np.ones(self.xMin.shape[0])
        self.currentCost = self.f(self.x)

        self.bestX = np.zeros(self.x.shape)
        self.bestCost = np.inf

        self.tempForParams = np.ones(self.x.shape)
        self.annealTimeForTemp = np.zeros(self.x.shape)

        self.acceptanceCostInitial = self.f(self.x)
        self.acceptanceCost = np.copy(self.acceptanceCostInitial)
        self.annealTimeForCost = 0

        self.c = c
        self.q = q
        self.numbReAnneal = numbReAnneal
        self.numbTempAnneal = numbTempAnneal
        self.numbIterations = numbIterations

    def generateNewPoint(self):
        newX = np.zeros(self.x.shape)
        
        for i,x_i in enumerate(self.x):
            while(True):
                u = random.random()
                temp = max(self.tempForParams[i],0.00000000000000001)
                dX = float(np.sign(u-0.5))*temp*((1.0+1.0/temp)**np.abs(2.0*u-1.0)-1.0)
                _x = x_i + dX*(self.xMax[i]-self.xMin[i])
                if(_x<self.xMax[i] and _x>self.xMin[i]):
                    break

            newX[i] = _x

        newCost = self.f(newX)

        logit = (self.currentCost-newCost)/float(self.acceptanceCost)
        pAccept = scipy.special.expit(logit)
        # assert self.f(self.x) == self.currentCost
        # pAccept = 1.0/(1.0+np.exp((newCost-self.currentCost)/self.acceptanceCost))

        if(random.random() <= pAccept):
            # print('\n'+str(self.currentCost)+' -> '+str(newCost)+'\n'+str(self.x)+' -> '+str(newX))
            # print('acceptanceCost: '+str(self.acceptanceCost))
            # print('pAccept: '+str(pAccept))
            return [True,newX,newCost]

        return [False,False,False]

    def findNewPoint(self):
        accepted,newX,newCost = self.generateNewPoint()
        if(accepted):
            self.x = np.copy(newX)
            self.currentCost = newCost
            if(self.currentCost < self.bestCost):
                self.bestX = np.copy(self.x)
                self.bestCost = self.currentCost
            return 1
        return 0

    def calculateSensitivities(self):
        DELTAX = 0.1
        sensitivities = np.zeros(self.x.shape)
        for i in range(sensitivities.shape[0]):
            one = np.zeros(self.bestX.shape)
            one[i] = DELTAX
            xOffset = self.bestX + one
            sensitivities[i] = np.abs(self.f(xOffset)-self.bestCost)/DELTAX
        return np.max(sensitivities)/sensitivities

    def reAnneal(self):
        sensitivityRatios = self.calculateSensitivities()
        # print('\n\nsensitivityRatios: '+str(sensitivityRatios))
        for i,ratio in enumerate(sensitivityRatios):
            self.tempForParams[i] *= ratio
            negOneC = -1.0/self.c
            logRatio = np.log(self.tempForParams[i])
            self.annealTimeForTemp[i] = max(1.0,(negOneC*logRatio)**self.x.shape[0])

            # print('\ni: '+str(i))
            # print('ratio: '+str(ratio))
            # print('self.tempForParams[i]: '+str(self.tempForParams[i]))
            # print('self.tempForParams[i]/ratio: '+str(self.tempForParams[i]/ratio))
            # print('initial: '+str(initial))
            # print('self.annealTimeForTemp[i]: '+str(self.annealTimeForTemp[i]))
            # assert self.tempForParams[i] < initial


        self.acceptanceCostInitial = self.currentCost
        self.acceptanceCost = self.bestCost
        negOneC = -1.0/self.c
        logRatio = np.log(self.acceptanceCost/float(self.acceptanceCostInitial))
        self.annealTimeForCost = max(1.0,(negOneC*logRatio)**self.x.shape[0])

    def tempAnneal(self):
        for i in range(self.x.shape[0]):
            self.annealTimeForTemp[i] += 1
            self.tempForParams[i] = np.exp(-self.c*self.annealTimeForTemp[i]**(self.q/float(self.x.shape[0])))

        self.annealTimeForCost += 1
        self.acceptanceCost = self.acceptanceCostInitial*np.exp(-self.c*self.annealTimeForCost**(self.q/float(self.x.shape[0])))

    def run(self):
        
        numbAccepted = 0
        count = 0
        while(True):

            numbAccepted += self.findNewPoint()

            if(count%self.numbTempAnneal == 0 and count > 0):
                self.tempAnneal()
            if(numbAccepted%self.numbReAnneal == 0 and numbAccepted > 0):
                self.reAnneal()
            if(count == self.numbIterations):
                break

            count += 1

            if(sum([x**2 for x in self.tempForParams]) < .00000000000000001):
                break

            # print('\n\nx: '+str(self.x))
            # print('currentCost: '+str(self.currentCost))
            # print('bestX: '+str(self.bestX))
            # print('bestCost: '+str(self.bestCost))
            # print('tempForParams: '+str(self.tempForParams))
            # print('annealTimeForTemp: '+str(self.annealTimeForTemp))
            # print('acceptanceCostInitial: '+str(self.acceptanceCostInitial))
            # print('acceptanceCost: '+str(self.acceptanceCost))
            # print('annealTimeForCost: '+str(self.annealTimeForCost))
            # print('numbAccepted: '+str(numbAccepted))
            # print('count: '+str(count))


        self.x = self.bestX
        self.currentCost = self.bestCost


        print('\n\nx: '+str(self.x))
        print('currentCost: '+str(self.currentCost))
        print('bestX: '+str(self.bestX))
        print('bestCost: '+str(self.bestCost))
        print('tempForParams: '+str(self.tempForParams))
        print('annealTimeForTemp: '+str(self.annealTimeForTemp))
        print('acceptanceCostInitial: '+str(self.acceptanceCostInitial))
        print('acceptanceCost: '+str(self.acceptanceCost))
        print('annealTimeForCost: '+str(self.annealTimeForCost))
        print('numbAccepted: '+str(numbAccepted))
        print('count: '+str(count))


def griewank(x):
    ans = 1.0
    prod = 1.0
    for i,_x in enumerate(x):
        ans += _x**2/4000.0
        prod *= np.cos(_x/np.sqrt(i+1))
    return ans - prod

def test():
    xMin = np.array([-10,-10,-10,-10,-10])
    xMax = np.array([10,10,10,10,10])
    c = 5.0
    q = 1.0
    numbReAnneal = 100
    numbTempAnneal = 1000
    numbIterations = 10000
    asa = ASA(griewank,xMin,xMax,c,q,numbReAnneal,numbTempAnneal,numbIterations)
    asa.run()

# test()












