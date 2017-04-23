"""
Establecer backend:
	https://xusite.wordpress.com/2016/02/23/setting-the-backend-of-keras/

Simulated Annealing:
	https://gist.github.com/wingedsheep/af2c630bc6fcdcfbef8450ff9689d45f

NeuPy: Neural Networks in Python
	http://neupy.com/pages/cheatsheet.html
"""
# Create first network with Keras
#import keras
import dataset
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def draw(data):
	import matplotlib.pyplot as plt
	from sklearn import datasets
	import matplotlib.pyplot as plt
	plt.plot(data) # history.history['loss']
	plt.show()


def Net:
	def __init__(self, net_input=1, opt='sgd', f_perdida='mse'):
		model = Sequential()
		model.add(Dense(50, input_dim=net_input, kernel_initializer='uniform', activation='sigmoid'))
		model.add(Dense(20, kernel_initializer='uniform', activation='sigmoid'))
		model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
		model.compile(loss=f_perdida, optimizer=opt, metrics=['accuracy'])

	def entrenar(self, red, x, y):
		n_epoch = 10
		c = []
		self.history = model.fit(x, y, epochs=n_epoch, batch_size=10, callbacks=c, verbose=0)

	def evaluar(self, x, y):
		scores = model.evaluate(x, y)
		print "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)

	def graficar(self):
		import matplotlib.pyplot as plt
		from sklearn import datasets
		import matplotlib.pyplot as plt
		plt.plot(self.history.history['loss']) # history.history['loss']
		plt.show()


def grupo1():
	model = Sequential()
	model.add(Dense(50, input_dim=2, kernel_initializer='uniform', activation='sigmoid'))
	model.add(Dense(20, kernel_initializer='uniform', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

	X, Y = dataset.dataset1()
	n_epoch = 10
	c = []
	history = model.fit(X, Y, epochs=n_epoch, batch_size=10, callbacks=c, verbose=0)

	# evaluate the model
	scores = model.evaluate(X, Y)
	print "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
	draw(history.history['loss'])

def grupo2():
	model = Sequential()
	model.add(Dense(50, input_dim=1, kernel_initializer='uniform', activation='sigmoid'))
	model.add(Dense(20, kernel_initializer='uniform', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

def grupo3():
	model = Sequential()
	model.add(Dense(50, input_dim=8, kernel_initializer='uniform', activation='sigmoid'))
	model.add(Dense(20, kernel_initializer='uniform', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

	X, Y = dataset.dataset3()
	n_epoch = 10
	c = []
	history = model.fit(X, Y, epochs=n_epoch, batch_size=10, callbacks=c, verbose=0)

	# evaluate the model
	scores = model.evaluate(X, Y)
	print "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
	draw(history.history['loss'])

grupo1()
"""
######################################
######################################
# fix random seed for reproducibility
DATAX, DATAY = dataset.dataset1()


######################################
######################################
# create model
model = Sequential()
model.add(Dense(50, input_dim=2, kernel_initializer='uniform', activation='sigmoid'))
model.add(Dense(20, kernel_initializer='uniform', activation='sigmoid'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


######################################
############################
# COMPILE MODEL
# Optimizer: https://keras.io/optimizers/
# loss: https://keras.io/losses/
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])


######################################
######################################
# Fit the model
n_epoch = 1000
c = []
#history = model.fit(X, Y, epochs=n_epoch, batch_size=10, callbacks=c)
history = model.fit(DATAX, DATAY, epochs=n_epoch, batch_size=10, callbacks=c, verbose=0)

import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.show()

# evaluate the model
scores = model.evaluate(DATAX, DATAY)
print "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
"""

"""
x, y [50, 20, 1]
x [50, 20, 1]
x [50, 20, 1]

x, y [100 1000 20]
"""


#############################
## https://matousc89.github.io/signalz/#
## https://matousc89.github.io/signalz/sources/generators/mackey_glass.html
#import signalz
#N = 1200
#x = signalz.mackey_glass(N, a=2.0, b=1.0, c=-1.0, d=2.0, e=9.65, initial=1.1)
#print x
##from keras.utils import plot_model
##plot_model(model, to_file='model.png')
