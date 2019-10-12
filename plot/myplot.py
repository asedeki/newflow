import random
import string
from math import pi

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D


def __couplage(g):
    Np = g.shape[0]
    v = 2 * np.pi / float(Np)
    gx = np.zeros(Np + 1)
    gy = np.zeros(Np + 1)
    k = 0
    for i in range(-Np // 2, Np // 2 + 1):
        gx[k] = i
        gy[k] = i
        k += 1
    # with open("test.dat", "w") as f:
    #     for i in range(-Np // 2, Np // 2 + 1):
    #         for j in range(-Np // 2, Np // 2 + 1):
    #             f.write(f"{i*v}\t {j*v}\t{g[i,-i,j]}\n")
    #         f.write("\n")
    gx, gy = np.meshgrid(gx, gy)
    gz = np.zeros(gx.shape, float)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            gz[i, j] = g[int(gx[i, j]), -int(gx[i, j]), int(gy[i, j])]

    return gx * v, gy * v, gz


def plot_pm3d(data, **kwargs):
    X, Y, Z = __couplage(data)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(np.min(Z), np.max(Z))
    ax.zaxis.set_major_locator(LinearLocator(4))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    plt.xticks(np.arange(-pi, pi + 0.01, pi / 2),
               ("$-\\pi$", "$-\\frac{\pi}{2}$",
                "0", "$\\frac{\pi}{2}$", "$\\pi$")
               )
    plt.yticks(np.arange(-pi, pi + 0.01, pi / 2),
               ("$-\\pi$", "$-\\frac{\pi}{2}$",
                "0", "$\\frac{\pi}{2}$", "$\\pi$")
               )
    if kwargs is not None:
        title = ""
        for k, v in kwargs.items():
            title += f"{k}={v} "
        plt.title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel("$k_\perp$")
    plt.ylabel("$k_\perp'$")
    letters = string.ascii_lowercase + '123456789'
    name = ''.join(random.choice(letters) for i in range(10))

    plt.savefig(f"{name}.pdf")
    # plt.show()
    plt.close()
    return f"{name}.pdf"


def plot_1d(x, y, *vargs, **kwargs):
    plt.plot(x, y, "g-o")
    if vargs is not None:
        legend = ""
        for k in vargs:
            legend += f"${k}$"
        plt.legend([legend])

    letters = string.ascii_lowercase + '123456789'
    name = ''.join(random.choice(letters) for i in range(10))
    plt.savefig(f"{name}.pdf")
    plt.close()
    return f"{name}.pdf"
