x = T.vector()
s = 1/(1+T.exp(-x))
ds = T.grad(T.sum(s), x) # Need sum to make s scalar

import matplotlib.pyplot as plt
%matplotlib inline

x0 = np.arange(-3, 3, 0.01).astype('float32')
plt.plot(x0, s.eval({x:x0}))
plt.plot(x0, ds.eval({x:x0}))

np.allclose(ds.eval({x:x0}), s.eval({x:x0}) * (1-s.eval({x:x0})))