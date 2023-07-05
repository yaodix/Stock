from shapesimilarity import shape_similarity
import matplotlib.pyplot as plt
import numpy as np
import similaritymeasures

x = np.linspace(1, -1, num=200)

y1 = 2*x**2 + 1
y2 = 2*x**2 + 2

shape1 = np.column_stack((x, y1))
shape2 = np.column_stack((x, y2))

# similarity = shape_similarity(shape1, shape2)
# df = similaritymeasures.frechet_dist(shape1, shape2)


plt.plot(shape1[:,0], shape1[:,1], linewidth=2.0)
plt.plot(shape2[:,0], shape2[:,1], linewidth=2.0)

plt.title(f'Shape similarity is: {df}', fontsize=14, fontweight='bold')
plt.show()