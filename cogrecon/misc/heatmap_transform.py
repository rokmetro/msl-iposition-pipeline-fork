from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib import mlab as ml
import numpy as np
from draggable_points import DraggablePoint
import matplotlib.patches as patches

circles = [patches.Circle((0, 0), 0.25, fc='r', alpha=0.5)]
drs = []

fig = plt.figure()
ax = fig.add_subplot(111)

n = 1e5
x = y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z1 = ml.bivariate_normal(X, Y, 2, 2, 0, 0)
z = Z1.ravel()
x = X.ravel()
y = Y.ravel()
gridsize = 30

hex_plt = plt.hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins=None)
plt.axis([x.min(), x.max(), y.min(), y.max()])


def update():
    global fig, ax, drs, hex_plt, x, y, z, gridsize
    Z = ml.bivariate_normal(X, Y, 2, 2, drs[0].point.center[0], drs[0].point.center[1])
    z = Z.ravel()
    hex_plt = plt.hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins=None)

for circ in circles:
    ax.add_patch(circ)
    dr = DraggablePoint(circ)
    dr.connect(update)
    drs.append(dr)

# anim = animation.FuncAnimation(fig, update, interval=10)

plt.show()
