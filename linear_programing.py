import numpy as np
import scipy.optimize


c = np.array([-3, -4], dtype=np.float64)
G = np.array([[1, 4], [2, 3], [2, 1]], dtype=np.float64)
h = np.array([1700, 1400, 1000], dtype=np.float64)

sol = scipy.optimize.linprog(c, A_ub=G, b_ub=h, bounds=(0, None))
print(sol.x)
print(sol.fun)
