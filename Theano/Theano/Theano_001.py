#import theano

import theano
from theano import tensor as T

# Define symbolic expression

a = T.scalar()
b = T.scalar()
y = a * b

mul = theano.function(inputs = [a, b], outputs = y)

multiply = mul(3, 2)
print(" ")
print(multiply)

