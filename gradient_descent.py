import numpy as np
import matplotlib.pyplot as plt


class GradientDescent(object):
    def __init__(self, f, df, alpha=0.01, eps=1e-6):
        """
        :param f: func will be minimized
        :param df: derivative func of f
        :param alpha: coefficient of update input vector
        :param eps: the criteria of extreme value
        """
        self.f = f
        self.df = df
        self.alpha = alpha
        self.eps = eps
        self.path = None

    def solve(self, initial_input):
        x = initial_input
        path = []  # transition of x
        grad = self.df(x)
        path.append(x)
        while (grad**2).sum() > self.eps ** 2:
            x = x - self.alpha * grad
            grad = self.df(x)
            path.append(x)

        self.path_ = np.array(path)
        self.x_ = x
        self.opt_ = self.f(x)



def f(xx):
    x = xx[0]
    y = xx[1]
    return 5 * x**2 - 6 * x*y + 3 * y**2 + 6 * x - 6 * y


def df(xx):
    x = xx[0]
    y = xx[1]
    return np.array([10*x - 6*y + 6, -6*x + 6*y - 6])


sample = GradientDescent(f, df)
sample.solve(np.array([1, 1]))

print(f'best x is {sample.x_}, minimum f(x) is {sample.opt_}')

path_x = sample.path_[:, 0]
path_y = sample.path_[:, 1]
plt.scatter(1, 1, color='k', marker='o')
plt.plot(path_x, path_y, color='k', linewidth=1.5)

contour_x = np.linspace(-2, 2, 200)
contour_y = np.linspace(-2, 2, 200)
z = f([contour_x, contour_y.reshape(-1, 1)])
levels=[-3, -2.9, -2.8, -2.6, -2.4, -2.2, -2, -1, 0, 1, 2, 3, 4]

plt.contour(contour_x, contour_y, z, levels=levels, colors='k', linestyles='dotted')
plt.show()