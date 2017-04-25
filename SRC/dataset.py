import theano
import theano.tensor as T
import numpy as np
from itertools import izip

import theano
import theano.tensor as T
from itertools import izip

# INPUT : 2
def dataset1(size=800):
	x = T.vector()
	y = T.vector()
	z = (np.sin(5*x*(3*y + 1)) + 1)/2
	f = theano.function([x, y], outputs=z)

	seed = 7
	np.random.seed(seed)
	a = np.random.rand(size).astype(np.float32)
	b = np.random.rand(size).astype(np.float32)
	aux = np.array(zip(a, b, f(a, b)))

	return aux[:,0:2], aux[:,2:].T[0], 2

def dataset2():
	t_final = 500
	a = 2.0
	b = 1.0
	m = 7.0
	delay = 2.0
	x_initial = 0.5

# INPUT : 8
def dataset3():
	dataset = np.loadtxt("cal_housing.data", delimiter=",")
	#X = dataset[:,0:8]
	#Y = dataset[:,8]
	#print X.shape, type(X), Y.shape, type(Y)
	return dataset[:,0:8], dataset[:,8], 8
