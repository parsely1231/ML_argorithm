import matplotlib.pyplot as plt
import numpy as np


class Newton(object):
    def __init__(self, f, df, eps=1e-10, calc_limit=1000):
        self.f = f
        self.df = df
        self.eps = eps
        self.calc_limit = calc_limit

    def solve(self, x0):
        calc = 0
        x = x0
        self.transition = x0.reshape(1, -1)
        while True:
            delta = np.dot(np.linalg.inv(self.df(x)), self.f(x))
            if (delta**2).sum() < self.eps**2:
                break
            x = x - delta
            self.transition = np.concatenate([self.transition, x.reshape(1, -1)])
            calc += 1
            if calc == self.calc_limit:
                print('The calculation limit has reached')
                break
        return x


def f1(x, y):
    return x**3 - 2*y


def f2(x, y):
    return x**2 + y**2 - 1


def f(xx):
    x = xx[0]
    y = xx[1]
    return np.array([f1(x, y), f2(x, y)])


def df(xx):
    x = xx[0]
    y = xx[1]
    df1 = [3*x**2, -2]
    df2 = [2*x, 2*y]
    return np.array([df1, df2])


x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
z1 = f1(x.reshape(1, -1), y.reshape(-1, 1))
z2 = f2(x.reshape(1, -1), y.reshape(-1, 1))

plt.ylim(-3, 3)
plt.xlim(-3, 3)

plt.contour(x, y, z1, colors='r', levels=[0])
plt.contour(x, y, z2, colors='k', levels=[0])

solver = Newton(f, df)

initials = [np.array([1, 1]), np.array([-1, -1]), np.array([2, -2])]
markers = ['+', '*', 'x']

for x0, mark in zip(initials, markers):
    ans = solver.solve(x0)
    plt.scatter(solver.transition[:, 0],
                solver.transition[:, 1], color='k', marker=mark)
    print(ans)

plt.show()