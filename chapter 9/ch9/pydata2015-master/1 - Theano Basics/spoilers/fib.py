a = theano.shared(1)
b = theano.shared(1)
f = a + b
updates = {a: b, b: f}
next_term = theano.function([], f, updates=updates)

[next_term() for _ in range(3, 10)]