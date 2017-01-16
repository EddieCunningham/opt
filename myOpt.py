import operator
import sys
import itertools
import scipy.optimize as opt
import numpy as np

def globalDescent(f):
	pass

def localDescent(f,x):
	res = opt.minimize(f,x)
	return res.x

def griewank(x):
	ans = 1
	prod = 1
	for i,_x in enumerate(x):
		ans += _x**2/4000
		prod *= np.cos(_x/np.sqrt(i+1))
	return ans - prod


x = np.array([1,1])
localDescent(griewank,x)