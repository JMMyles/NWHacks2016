import numpy as np

c1,c2 = 5.0,2.0
x = np.arange(1,11)/10.0
y = c1*np.exp(-x)+c2*x
b = y + 0.01*max(y)*np.random.randn(len(y))
A = np.column_stack((np.exp(-x),x))
c,resid,rank,sigma = np.linalg.lstsq(A,b)
print(c)
# [ 4.96579654  2.03913202]