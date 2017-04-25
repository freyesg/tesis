import dataset
import sannealing as sa
import neural_network as nn
## ENTRADAS : 8
#x, y, size = dataset.dataset3()
# ENTRADAS : 2
x, y, size = dataset.dataset1()

SOLUCIONES, T_0, cte = 2, 50, 0.9
#SOL_ARRAY = [2, 5, 10, 15]
#R_ARRAY = [15, 20, 30, 40]
SOL_ARRAY = [15]
R_ARRAY = [20]

print "SIMULATED ANNEALING"
#print "(soluciones, radio, error)"
for s in SOL_ARRAY:
	for r in R_ARRAY:
		#(soluciones, radio, error)
		#print ("%4.0i\t%4.10f\t")%(s, r),
		sanet = sa.SANet(input=size, solutions=s, T=T_0, c=cte)
		print sanet.entrenar(x, y, r)
print
sanet.dibujar()
sanet = None

print "SGD"
net = nn.Net(net_input=size)
print net.entrenar(x, y, verb=0)

net.dibujar()
