import neural_network as nn

class SetNet():
	def __init__(self, size = 5):
		self.n = size

	def random(self):
		a = []

		for i in range(self.n):
			a.append(nn.Net(net_input=self.input))
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
			n = nn.Net(net_input=input)
			n.set_weights(w)
			a.append(n)
			o = (o + theta)%360
		return a

if __name__ == "__main__":
	import dataset
	## ENTRADAS : 8
	#x, y, size = dataset.dataset3()
	# ENTRADAS : 2
	x, y, size = dataset.dataset1()

	bla = SetNet(3)
	n = nn.Net(net_input=size, i=1)
	print bla.rotate(n, 20, size)
