import theano
import theano.tensor as T
import numpy as np
from itertools import izip

import theano
import theano.tensor as T
from itertools import izip

# INPUT		: 2
# OUTPUT	: 1
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

# http://www.scholarpedia.org/article/Mackey-Glass_equation
# http://www.mathworks.com/matlabcentral/fileexchange/24390-mackey-glass-time-series-generator?focused=5119961&tab=example
# http://organic.elis.ugent.be/oger
# https://github.com/npinto/Oger
def dataset2(samples_n = 2424, y = 1, b = 2, n = 9.65, r = 2):
	data = np.zeros(samples_n)
	for idx, i in enumerate(data):
	    if idx < r:
	        data[idx] = 1.1 + 0.1 * i
	    else:
	        xr = data[idx - r]
	        x = data[idx - 1]

	        data[idx] = x + (b * xr / (1 + xr**n) - y * x)

	# NORMALIZAR DATOS
	minimo, maximo = min(data), max(data)
	diff = maximo - minimo
	for idx, i in enumerate(data):
		data[idx] = (data[idx] - minimo)/diff

	i = 24
	x = []
	y = []

	while i < len(data):
		x.append([data[i - 24], data[i - 18], data[i - 12], data[i - 6]])
		y.append(data[i])
		i += 1
	x = np.array(x)
	y = np.array(y)

	return x, y, 4


# INPUT		: 8
# OUTPUT	: 1
def dataset3():
	dataset = np.loadtxt("cal_housing.data", delimiter=",")
	minimo = dataset[0]
	maximo = dataset[0]
	for idx, i in enumerate(dataset):
		minimo = np.minimum(minimo, i)
		maximo = np.maximum(minimo, i)

	diff = np.subtract(maximo, minimo)
	for idx, i in enumerate(dataset):
		dataset = np.divide(np.subtract(dataset, minimo), diff)

	return dataset[:,0:8], dataset[:,8], 8

if __name__ == "__main__":
	print "set1", dataset1()
