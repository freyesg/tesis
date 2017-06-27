from time import time
import matplotlib.pyplot as plt
import datetime
import neural_network as nn
import random as rnd
import setnet as sn
import numpy as np

class SANet():
	def __init__(self, input=1, solutions=1, T=50, c=0.9, i=0):
		self.i = i

		self.input = input
		self.net = nn.Net(net_input=self.input, i=self.i)
		self.n_sol = solutions
		self.T = T
		self.c = c
		self.error_array = []

	def get_trainable_params(self):
		"""
		params = []
		for layer in self.model.layers:
			params.append(layer.get_weights())
		return params
		"""
		return self.net.get_trainable_params()

	"""
	def solution_set(self):
		s = []
		for i in range(self.n_sol):
			s.append(nn.Net(net_input=self.input, i=self.i))
		return s
	"""

	def solution_set(self, radio=20):
		"""
		#np.random.seed(datetime.datetime.now().microsecond)
		s = []

		for i in range(self.n_sol):
			aux = nn.Net(net_input=self.input, i=self.i)
			aux.set_weights(
				aux.rotate(
					np.radians(np.random.uniform(0, 360)),
					#self.net,
					r=np.random.uniform(0, radio)
				)
			)
			s.append(aux)
		"""
		np.random.seed(datetime.datetime.now().microsecond)
		s = []
		giro = 0
		for i in range(self.n_sol):
			aux = nn.Net(net_input=self.input, i=self.i)
			#giro = np.random.uniform(0, 360)
			giro = giro + 360.0/self.n_sol
			w = self.net.rotate(
				np.radians(giro),
				#self.net,
				r=np.random.uniform(0, radio)
			)
			aux.set_weights(w)
			s.append(aux)
		return s


	def entrenar(self, x, y, T=None, c=None, radio=20):
		if T != None:
			self.T = T
		if c != None:
			self.c = c
		T = self.T
		c = self.c
		#print x, y
		k = 1.38065*10**(-23) #J/K

		contador = 0
		#x_0, y_0 = np.array([x[0]]), np.array([y[0]])
		x_0, y_0 = x, y
		t_i = time()
		self.error_array = []
		while T >= 0.01:
			#print "%4.2i - T(%4.10f):"%(contador, T), self.net.evaluar(x_0, y_0)
			"""
			np.random.seed(datetime.datetime.now().microsecond)
			s = []
			for i in range(self.n_sol):
				aux = nn.Net(net_input=self.input, i=self.i)
				w = self.net.rotate(
					np.radians(np.random.uniform(0, 360)),
					self.net,
					r=np.random.uniform(0, radio)
				)
				aux.set_weights(w)
				s.append(aux)
			"""
			s = self.solution_set(radio)

			for i in range(self.n_sol):
				aux = rnd.randint(0, len(s)-1)
				new, old = s[aux].evaluar(x_0, y_0)[0], self.net.evaluar(x_0, y_0)[0]
				d = new - old
				r = rnd.uniform(0, 1)
				bla = -d/k*T

				if d <= 0:
					self.net = s[aux]
				elif r < bla:
					self.net = s[aux]
				del s[aux]

			T *= c
			contador += 1
			self.error_array.append(self.net.evaluar(x_0, y_0)[0])
		t_f = time()
		return self.net.evaluar(x_0, y_0), t_f - t_i

	def evaluar(self, x, y):
		if len(x) > 1:
			scores = self.net.evaluar(x, y)
			n = len(scores)
			err = sum(scores)
			scores = err/n
		else:
			scores = self.net.evaluar(x, y)
		return scores

	def predecir(self, x, b_s=32, v=0):
		return self.net.predecir(x, batch_size=b_s, verbose=v)

	def graficar(self):
		plt.plot(self.error_array) # history.history['loss']
		plt.show()

	def get_data(self):
		return self.error_array

	def dibujar(self, filename="nnsa"):
		plt.plot(self.error_array)
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train'], loc='upper left')
		plt.savefig(filename+'_loss'+'.png', bbox_inches='tight')

if __name__ == "__main__":
	import dataset
	## ENTRADAS : 8
	#x, y, size = dataset.dataset3()
	# ENTRADAS : 2
	x, y, size = dataset.dataset1()


	print "SIMULATED ANNEALING"
	san = SANet(input=size, solutions=5, i=1)
	#san.entrenar(x, y)
	for i in san.solution_set():
		print i.get_trainable_params()
		print
