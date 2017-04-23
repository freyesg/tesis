"""
Establecer backend:
https://xusite.wordpress.com/2016/02/23/setting-the-backend-of-keras/

Simulated Annealing:
https://gist.github.com/wingedsheep/af2c630bc6fcdcfbef8450ff9689d45f

CNNs with Keras:
http://euler.stat.yale.edu/~tba3/stat665/lectures/lec17/notebook17.html
"""
# Create first network with Keras
import neural_network as nn
import sannealing as sa
import dataset
#import random as rnd
import math
import numpy as np
import theano
import matplotlib.pyplot as plt
from numpy import linalg as LA
import setnet as sn

"""
class SetNet():
	def __init__(self, size = 5):
		self.n = size

	def random(self):
		a = []

		for i in range(self.n):
			a.append(nn.Net(net_input=self.input, i=1))
		return a

	def scale(self, net):
		pass
		AUX = net.get_trainable_params()
		for aux in AUX[0]:
			for i in aux:
				print i/LA.norm(i) # Vector normalizado
			print

	def rotate(self, net, theta=20, input=2):
		a = []
		o = theta
		for i in range(self.n):
			w = net.rotate(theta)
			n = nn.Net(net_input=input, i=1)
			n.set_weights(w)
			a.append(n)
			o = (o + theta)%360
		return a
"""



## ENTRADAS : 8
#x, y, size = dataset.dataset3()
# ENTRADAS : 2
x, y, size = dataset.dataset1()

bla = sn.SetNet(2)
n = nn.Net(net_input=size, i=1)
bla.rotate(n, 20, size)

#############################
## https://matousc89.github.io/signalz/#
## https://matousc89.github.io/signalz/sources/generators/mackey_glass.html
#import signalz
#N = 1200
#x = signalz.mackey_glass(N, a=2.0, b=1.0, c=-1.0, d=2.0, e=9.65, initial=1.1)
#print x
##from keras.utils import plot_model
##plot_model(model, to_file='model.png')
