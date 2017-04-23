import theano
import numpy as np
import numpy.random as npr

n = 4
x = theano.tensor.matrix('x')
y = theano.tensor.vector('y')

f = theano.function([y, x], theano.dot(y, x))

I = np.identity(n)
a = npr.randn(n).astype(theano.config.floatX)

print a
print I
print f(a, I)
