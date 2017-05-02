import matplotlib.pyplot as plt
import dataset
import sannealing as sa
import neural_network as nn
import csv

## ENTRADAS : 8
#x, y, size = dataset.dataset2()
# ENTRADAS : 2
#x, y, size = dataset.dataset1()

SOLUCIONES, T_0, cte = 2, 50, 0.9
"""
#SOL_ARRAY = [2, 5, 10, 15]
#R_ARRAY = [15, 20, 30, 40]
SOL_ARRAY = [15]
R_ARRAY = [20]
for s in SOL_ARRAY:
	for r in R_ARRAY:
		#(soluciones, radio, error)
		sanet = sa.SANet(input=size, solutions=s, T=T_0, c=cte)
		print sanet.entrenar(x, y, r)
"""

class Data:
	def __init__(self, d=[], name=""):
		self.data = d
		self.name = name

	def write(self, f):
		f = open(self.name + f + ".dat", 'w')
		writer = csv.writer(f)
		writer.writerow(self.data)


def dibujar(DATA, title="Model loss", filename="plot"):
	label = []
	for i in DATA:
		plt.plot(i.data)
		label.append(i.name)

	plt.title(title)
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(label, loc='upper right')
	plt.savefig(filename+'.png', bbox_inches='tight')
	plt.clf()
	plt.cla()
	plt.close()

n = 10
i = 0
for x, y, size in [dataset.dataset1(), dataset.dataset2(), dataset.dataset3()]:
	for j in range(n):
		##print "SIMULATED ANNEALING".
		##(soluciones, radio, error)
		sanet = sa.SANet(input=size, solutions=15, T=T_0, c=cte)
		sanet.entrenar(x, y, 20)

		##print "SGD"
		net = nn.Net(net_input=size)
		net.entrenar(x, y, verb=0)

		s = "_"+ str(i) +"_"+ str(j)
		r = [
			Data(sanet.get_data(), "SA"),
			Data(net.get_data(), "SGD")
		]
		for k in r:
			k.write("data"+ s)
		dibujar(r, filename="plot"+ s)
	i += 1
