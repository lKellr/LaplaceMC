from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np

data = np.loadtxt("results500.csv", delimiter=",", dtype=np.double)

fig, ax = plt.subplots()
im = ax.imshow(
    data, cmap="viridis", vmin=-1.0, vmax=1.0, origin="lower", extent=(-1, 1, -1, 1)
)
circle_lowerleft = Circle(
    (-1, -1),
    2.0 / np.sqrt(2),
    facecolor="none",
    edgecolor="black",
    linewidth=2,
)

circle_upperight = Circle(
    (1, 1),
    2.0 / np.sqrt(2),
    facecolor="none",
    edgecolor="black",
    linewidth=2,
)
ax.add_patch(circle_lowerleft)
ax.add_patch(circle_upperight)

fig.colorbar(im, ax=ax)
# fig.savefig("results500.svg")
fig.show()
