# https://www.datatechnotes.com/2021/11/b-spline-fitting-example-in-python.html

from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np 

y = [0, 1, 3, 4, 3, 5, 7, 5, 2, 3, 4, 8, 9, 8, 7]
n = len(y)
x = range(0, n) 

tck = interpolate.splrep(x, y, s=0, k=3) 
x_new = np.linspace(min(x), max(x), 100)
y_fit = interpolate.BSpline(*tck)(x_new)

plt.title("BSpline curve fitting")
plt.plot(x, y, 'ro', label="original")
plt.plot(x_new, y_fit, '-c', label="B-spline")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid()
plt.show() 