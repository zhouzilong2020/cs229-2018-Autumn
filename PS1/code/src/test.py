import numpy as np

a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
a = a.T
b = np.array([2,2,2,2])
b = b.T
print(a * b)
print(a @ b)
