import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from math import pi
import os
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def couplage(g):
    Np = g.shape[0]
    v = 2 * np.pi / float(Np)
    n = (Np + 1)**2
    gx = np.zeros(n)
    gy = np.zeros(n)
    gz = np.zeros(n)
    k = 0
    line = ""
    for i in range(-Np // 2, Np // 2 + 1):
        if k != 0:
            line += "\n"
        for j in range(-Np // 2, Np // 2 + 1):
            gx[k] = i * v
            gy[k] = j * v
            gz[k] = g[i, -i, j]
            k += 1
    return gx, gy, gz


def plot3d(T, gx, gy, gz):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(-pi, pi + 0.1, 0.1)
    y = np.arange(-pi, pi + 0.1, 0.1)
    X, Y = np.meshgrid(gx, gy)
    Z = griddata((gx, gy), gz, (x[None, :], y[:, None]), method='cubic')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    # plt.savefig(f"pg{T}.pdf")
    # plt.close()


def plot(T, gx, gy, gz):
    x = np.arange(-pi, pi + 0.1, 0.1)
    y = np.arange(-pi, pi + 0.1, 0.1)
    # method=cubic
    zi = griddata((gx, gy), gz, (x[None, :], y[:, None]), method='bicubic')
    plt.imshow(zi, origin="lowwer", vmin=0, cmap="hot",
               aspect="auto", extent=[-pi, pi, -pi, pi])
    plt.xlabel("$k_\perp$")
    plt.ylabel("$k_\perp'$")
    plt.title(f"$T={T}$")
    plt.xticks(np.arange(-pi, pi + 0.01, pi / 2),
               ("$-\pi$", "$-\pi/2$", "0", "$\pi/2$", "$\pi$"))
    plt.colorbar()
    # plt.show()
    plt.savefig(f"pg{T}.pdf")
    plt.close()


def plotN(T, gx, gy, gz):
    halfpurples = {'blue': [(0.0, 1.0, 1.0), (0.000001, 0.78431373834609985, 0.78431373834609985),
                            (0.25, 0.729411780834198, 0.729411780834198),
                            (0.5, 0.63921570777893066, 0.63921570777893066),
                            (0.75, 0.56078433990478516, 0.56078433990478516),
                            (1.0, 0.49019607901573181, 0.49019607901573181)],

                   'green': [(0.0, 1.0, 1.0), (0.000001, 0.60392159223556519, 0.60392159223556519),
                             (0.25, 0.49019607901573181, 0.49019607901573181),
                             (0.5, 0.31764706969261169, 0.31764706969261169),
                             (0.75, 0.15294118225574493, 0.15294118225574493), (1.0, 0.0, 0.0)],

                   'red': [(0.0, 1.0, 1.0), (0.000001, 0.61960786581039429, 0.61960786581039429),
                           (0.25, 0.50196081399917603, 0.50196081399917603),
                           (0.5, 0.41568627953529358, 0.41568627953529358),
                           (0.75, 0.32941177487373352, 0.32941177487373352),
                           (1.0, 0.24705882370471954, 0.24705882370471954)]}

    halfpurplecmap = mpl.colors.LinearSegmentedColormap(
        'halfpurples', halfpurples, 256)

    x = np.arange(-pi, pi, 0.05)
    y = np.arange(-pi, pi, 0.05)
    zi = griddata((gx, gy), gz, (x[None, :], y[:, None]), method='cubic')
    # cmaptype = "hot"  # 'jet'
    cmaptype = halfpurplecmap
    aspectType = "auto"  # "equal"
    limits = (-pi, pi, -pi, pi)
    plt.imshow(zi, origin="lower", vmin=0, interpolation="bicubic", cmap=cmaptype, aspect=aspectType,
               extent=limits)  # ,interpolation='bicubic'
    title = f"T = {T}"
    plt.colorbar()
    plt.xlabel(r"$k_{\perp}$")
    plt.ylabel(r"$k_{\perp'}$")
    plt.title(r"$T=%s$ , $g(k_{\perp},-k_{\perp},k_{\perp'})$" % T)
    plt.savefig(f"pg{T}.pdf")
    plt.close()


if __name__ == "__main__":
    path = pathlib.Path()
    files = path.rglob('*.npy')
    result = {}
    for f in files:
        data = np.load(f, allow_pickle=True)[()]
        result = data
        #result[float(r["T"])] = {"g1": r["g1"], "g2": r["g2"], "g3": r["g3"]}
    Temp = list(result.keys())
    # Temp.sort()
    #Temp = Temp[-1::-1]
    pdfs = []
    for T in Temp:
        if isinstance(T, float):
            try:
                pdfs.append(f"pg{T}.pdf")
                g = result[T][2]  # result[T]["g1"]+result[T]["g2"]
                gx, gy, gz = couplage(g)
                print(np.sum(np.abs(g)))
                plot3d(T, gx, gy, gz)
            except Exception as e:
                print(e)
                print(f"T={T}")
    pdf = " ".join(pdfs)
    os.system(f"pdfunite {pdf} out.pdf")
    os.system(f"rm -f {pdf}")
