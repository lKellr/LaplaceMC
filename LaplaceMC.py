# Implementation of a Monta-Carlo solver for the Laplace equation, inspired by Körner "Fourier Analysis" chapter 13

import numpy as np
from matplotlib import pyplot as plt

# parameters
n_nodes = 100
samples_per_cell = 5
boundary_func = lambda x, y: np.sqrt(
    np.maximum(
        np.maximum(0.5 - x**2 - y**2, 0.5 - (x - 1.0) ** 2 - (y - 1.0) ** 2), 0.0
    )
)  # NOTE: must be less than one in magnitude for all values on the boundary
boundary_geometry_lvs = lambda x, y: np.minimum(
    x**2 + y**2 - 0.5, (x - 1.0) ** 2 + (y - 1.0) ** 2 - 0.5
)

# random generator
rng = np.random.default_rng(12345)

# create grid
x = np.linspace(0.0, 1.0, n_nodes)
y = np.linspace(0.0, 1.0, n_nodes)

phi = np.zeros((n_nodes, n_nodes))
phi[0, :] = boundary_func(x[0], y)
phi[-1, :] = boundary_func(x[-1], y)
phi[:, 0] = boundary_func(x, y[0])
phi[:, -1] = boundary_func(x, y[-1])

# run simulation
for ix_n in range(samples_per_cell):
    for ix_i in range(1, n_nodes - 1):
        for ix_j in range(1, n_nodes - 1):
            ix_i_particle, ix_j_particle = ix_i, ix_j
            while (
                ix_i_particle != 0
                and ix_i_particle != n_nodes - 1
                and ix_j_particle != 0
                and ix_j_particle != n_nodes - 1
                and boundary_geometry_lvs(x[ix_i_particle], y[ix_j_particle]) < 0.0
            ):
                rand_vec = rng.integers(2, size=2) * 2 - 1
                ix_i_particle += rand_vec[0]
                ix_j_particle += rand_vec[1]
            phi[ix_i, ix_j] += (
                boundary_func(x[ix_i_particle], y[ix_j_particle]) - phi[ix_i, ix_j]
            ) / (ix_n + 1)
    # print(np.sum((np.diff(phi, 2, 0)[:, 1:-1] + np.diff(phi, 2, 1)[1:-1, :]) ** 2))

# plot
# X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots()

ax.imshow(phi, cmap="viridis", vmin=-1.0, vmax=1.0, origin="lower")

plt.show()
