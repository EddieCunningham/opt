import operator
import sys
import itertools
import scipy.optimize as opt
import random
import scipy.special
import numpy as np

class ASA():

    def __init__(self,f,parameterBounds,c,q,numbRe,numbTemp,numbIt):
        self.parameters = np.random.random(parameterBounds[0].shape)
        self.parameterBounds = parameterBounds
        self.f = f

        self.paramTemperatures = 10*np.ones(self.parameters.shape)
        self.tempCost = self.f(self.parameters)
        self.kCost = 0
        self.tempCost_0 = self.tempCost
        self.paramAnnealTimes = np.zeros(self.parameters.shape)
        self.bestParams = np.copy(self.parameters)
        self.numbAccepted = 0
        self.energy = np.inf
        self.bestEnergy = np.inf

        self.c = c
        self.q = q
        self.numbRe = numbRe
        self.numbTemp = numbTemp
        self.numbIt = numbIt


    def updateParameters(self):
        potentialNewParameters = np.zeros(self.parameters.shape)

        for i in range(self.parameters.shape[0]):
            while(True):
                if(np.abs(self.paramTemperatures[i]) < 10**-35):
                    potentialNewParameters[i] = self.parameters[i]
                u = random.random()
                sign = np.sign(u-0.5)
                middle = (1.0 + 1.0/self.paramTemperatures[i])**np.abs(2.0*u - 1.0) - 1.0
                dParam = sign*self.paramTemperatures[i]*middle
                potentialNewParameters[i] = self.parameters[i] + dParam*(self.parameterBounds[1][i] - self.parameterBounds[0][i])
                if(potentialNewParameters[i] < self.parameterBounds[1][i] and potentialNewParameters[i] > self.parameterBounds[0][i]):
                    break

        newEnergy = self.f(potentialNewParameters)
        logit = (self.energy-newEnergy)/float(self.tempCost)
        pAccept = scipy.special.expit(logit)

        if(pAccept >= random.random()):
            self.parameters = potentialNewParameters
            self.energy = newEnergy
            return True
        return False

    def reAnneal(self):
        sensitivities = np.zeros(self.parameters.shape)

        for i in range(self.parameters.shape[0]):
            newParameters = np.copy(self.parameters)
            newParameters[i] += 0.01

            sensitivities[i] = abs(self.f(newParameters) - self.energy)*(self.parameterBounds[1][i] - self.parameterBounds[0][i])

        sensitivities = max(sensitivities)/sensitivities

        for ratio in sensitivities:
            self.paramTemperatures[i] *= ratio
            changing = (-1.0/self.c*np.log(self.paramTemperatures[i]))**self.parameters.shape[0]

        self.tempCost = self.bestEnergy
        self.tempCost_0 = self.energy
        self.kCost = (-1.0/self.c*np.log(self.tempCost/self.tempCost_0))**self.parameters.shape[0]

    def temperatureAnneal(self):
        for i in range(self.paramTemperatures.shape[0]):
            self.paramAnnealTimes[i] += 1
            self.paramTemperatures[i] = np.exp(-self.c*(self.paramAnnealTimes[i])**(self.q/self.paramTemperatures.shape[0]))

        self.kCost += 1
        self.tempCost = self.tempCost_0*np.exp(-self.c*(self.kCost)**(self.q/self.paramTemperatures.shape[0]))

    def run(self):

        for i in range(self.numbIt):
            # print('\n\ni: '+str(i))
            # print('parameters: '+str(self.parameters))
            # print('parameterBounds: '+str(self.parameterBounds))
            # print('f: '+str(self.f))
            # print('paramTemperatures: '+str(self.paramTemperatures))
            # print('tempCost: '+str(self.tempCost))
            # print('kCost: '+str(self.kCost))
            # print('tempCost_0: '+str(self.tempCost_0))
            # print('paramAnnealTimes: '+str(self.paramAnnealTimes))
            # print('bestParams: '+str(self.bestParams))
            # print('numbAccepted: '+str(self.numbAccepted))
            # print('energy: '+str(self.energy))
            # print('bestEnergy: '+str(self.bestEnergy))
            # print('c: '+str(self.c))
            # print('q: '+str(self.q))
            # print('numbRe: '+str(self.numbRe))
            # print('numbTemp: '+str(self.numbTemp))
            # print('numbIt: '+str(self.numbIt))


            accepted = self.updateParameters()
            if(accepted):
                self.numbAccepted += 1
                if(self.energy < self.bestEnergy):
                    self.bestParams = np.copy(self.parameters)
                    self.bestEnergy = self.energy

            if(self.numbAccepted == self.numbRe):
                self.reAnneal()
                self.numbAccepted = 0

            if(i > 0 and i%self.numbTemp == 0):
                self.temperatureAnneal()

        print('\n\ni: '+str(i))
        print('parameters: '+str(self.parameters))
        print('parameterBounds: '+str(self.parameterBounds))
        print('f: '+str(self.f))
        print('paramTemperatures: '+str(self.paramTemperatures))
        print('tempCost: '+str(self.tempCost))
        print('kCost: '+str(self.kCost))
        print('tempCost_0: '+str(self.tempCost_0))
        print('paramAnnealTimes: '+str(self.paramAnnealTimes))
        print('bestParams: '+str(self.bestParams))
        print('numbAccepted: '+str(self.numbAccepted))
        print('energy: '+str(self.energy))
        print('bestEnergy: '+str(self.bestEnergy))
        print('c: '+str(self.c))
        print('q: '+str(self.q))
        print('numbRe: '+str(self.numbRe))
        print('numbTemp: '+str(self.numbTemp))
        print('numbIt: '+str(self.numbIt))

        self.parameters = self.bestParams
        return self.bestEnergy


def griewank(x):
    ans = 1.0
    prod = 1.0
    for i,_x in enumerate(x):
        ans += _x**2/4000.0
        prod *= np.cos(_x/np.sqrt(i+1))
    return ans - prod

def test():
    parameterBounds = np.array([[-10,-10,-10,-10,-10],
                     [ 10, 10, 10, 10, 10]])
    c = 5.0
    q = 1.0
    numbRe = 100
    numbTemp = 1000
    numbIt = 100000
    asa = ASA(griewank,parameterBounds,c,q,numbRe,numbTemp,numbIt)
    asa.run()

test()





