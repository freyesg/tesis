import numpy as np
import datetime
from time import time
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense
import theano
from numpy import linalg as LA



class Net:
	def __init__(self, net_input=1, net_output=1, opt='sgd', f_perdida='mse', i=0, j=0):
		np.random.seed(datetime.datetime.now().microsecond)

		self.model = Sequential()
		#[50 20 1]
		L1=[50, 3]
		L2=[20, 2]
		kernel = [keras.initializers.RandomNormal(), 'random_uniform']
		self.model.add(Dense(L1[i], input_dim=net_input,	kernel_initializer=kernel[j], activation='sigmoid', bias_initializer='zeros'))
		self.model.add(Dense(L2[i], 						kernel_initializer=kernel[j], activation='sigmoid', bias_initializer='zeros'))
		self.model.add(Dense(net_output,					kernel_initializer=kernel[j], activation='sigmoid', bias_initializer='zeros'))

		self.model.compile(loss=f_perdida, optimizer=opt, metrics=['accuracy'])
		self.h = None

	def entrenar(self, x, y, verb=0, n_epoch=100):
		call = []
		t_i = time()
		self.h = self.model.fit(x, y, epochs=n_epoch, batch_size=10, callbacks=call, verbose=verb)
		t_f = time()
		return self.evaluar(x, y), t_f - t_i

	def evaluar(self, x, y, v=0):
		if len(x) > 1:
			scores = self.model.evaluate(x, y, verbose=v)
			n = len(scores)
			err = sum(scores)
			scores = err/n
		else:
			scores = self.model.evaluate(x, y, verbose=v)
		return scores

	def predecir(self, x, b_s=32, v=0):
		return self.model.predict(x, batch_size=b_s, verbose=v)

	def graficar(self):
		plt.plot(self.h.history['loss']) # history.history['loss']
		plt.show()

	def dibujar(self):
		plt.plot(self.h.history['loss']) # history.history['loss']
		plt.savefig("nn.png", bbox_inches='tight')

	def set_weights(self, w):
		for l, layer in enumerate(self.model.layers):
			self.model.layers[l].set_weights(w[l])

	def get_trainable_params(self):
		params = []
		for layer in self.model.layers:
			params.append(layer.get_weights())
		return params

	def get_weights(self):
		params = []
		for layer in self.model.layers:
			params.append(layer.get_weights())
		return params

	#x_rotated = ((x - x_origin) * cos(angle)) - (-(y - y_origin) * sin(angle)) + x_origin
	#y_rotated = (-(y - y_origin) * cos(angle)) - ((x - x_origin) * sin(angle)) + y_origin
	def scale(self, s=1):
		weights = []
		AUX = self.get_trainable_params()

		for idx, b in enumerate(AUX):
			i, bias = b
			c = []
			c.append(i.dot(s).astype("float32"))
			c.append(bias)
			weights.append(c)

		return weights

	def translate(self, r=1):
		weights = []
		AUX = self.get_trainable_params()

		for idx, b in enumerate(AUX):
			i, bias = b

	def rotate(self, theta, O=None, r=1):
		weights = []
		AUX = self.get_trainable_params()

		if O != None:
			O = O.get_trainable_params()
			for i in range(len(O)):
				for j in range(len(O[i][0])):
					AUX[i][0][j] = AUX[i][0][j] - O[i][0][j]

		for idx, b in enumerate(AUX):
			i, bias = b
			k, l = np.random.randint(len(i[0]), size=2)
			I = np.identity(len(i[0]))
			while k == l and len(i[0]) != 1: k = np.random.randint(len(i[0]))
			if l < k:
				aux = l
				l = k
				k = aux
			I[k][k], I[k][l] = r*np.cos(theta), -r*np.sin(theta)
			I[l][k], I[l][l] = r*np.sin(theta), r*np.cos(theta)

			c = []
			c.append(i.dot(I).astype("float32"))
			c.append(bias)
			weights.append(c)

		if O != None:
			for i in range(len(O)):
				for j in range(len(O[i][0])):
					weights[i][0][j] = weights[i][0][j] + O[i][0][j]
		return weights


if __name__ == "__main__":
	import dataset
	## ENTRADAS : 8
	#x, y, size = dataset.dataset3()
	# ENTRADAS : 2
	x, y, size = dataset.dataset1()

	print "SGD"
	n = Net(net_input=size, i=1)
	n.entrenar(x, y, verb=0)
	x_0, y_0 = x, y
	print n.evaluar(x_0, y_0)
	print n.get_trainable_params()
	print
	print n.rotate(90)
