# LaplaceMC
A stochastic method for solving Laplace's equation on complicated domains with Dirichlet boundary conditions.

# Codes
## LaplaceMC.py
Python code that demonstrates the algorithm and is easy to modify. Python is however much too slow for these loops over single cells without vectorization possibilities.

Allows for the specification of a level-set function. With this, complicated domain geometries are possible. The level-set function is normally $$\psi(\vec{x}, t) = \begin{cases} -&\operatorname{dist}(\vec{x}, \Gamma) &\text{ if } \vec{x} \in \Omega \\ &\operatorname{dist}(\vec{x}, \Gamma)  &\text{ otherwise } \end{cases}$$ for a domain $\Omega$ with boundary $\Gamma$, but it is sufficient if it takes negative values inside $\Omega$ and positive values outside.

Note also that boundary functions are specified by a function $g(\vec{x})$ defined on the whole domain $\Omega$, not just $\partial  \Omega$. This is required as nodes outside $\Omega$ usually do not lie exactly on $\partial\Omega$.

## LaplaceMClvs.cpp
This c++ code calculates the solution much faster on the square $x, y \in [-1, 1]$.

The solution is output in a .csv file called results{number of nodes}{smoothing?}.csv. It can be plotted by the provided plot.py script.
Parallelized with OpenMP.

Optionally, some iterations of a smoothing kernel can be applied to the result. Variants:
1) central finite differences
2) 3x3 mean filter
3) 3x3 Gaussian filter
All smoothers should converge to the true solution on their own.
