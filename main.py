import numpy as np
import ProblemDefinition as problem_def
import FemSolver as solver
from dolfin import *
import visualiser as vs
import analyser as analyser
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(20, 20)
g_x = "pow(x[0], 6)"
g_y = "pow(x[1], 6)"
g = Expression((g_x, g_y), degree=2)
lin_elas = problem_def.LinearElasticity(g)
solver = solver.FemSolver()
u_sol = solver.solve(lin_elas, mesh, 1)
plot(u_sol)
plt.show()


"""
visualiser = vs.Visualiser()
u0 = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=6)
f = Expression("(8*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2)
helmholtz = problem_def.HelmholtzProblem(f, u0)

polynomial_degrees = [1, 2]
step_sizes = np.logspace(np.log10(np.sqrt(0.001)), np.log10(np.sqrt(0.1)), num=4)
analyser = analyser.Analyser(helmholtz)
results = analyser.run_analysis(polynomial_degrees, step_sizes, u0)
visualiser.convergence_rates_plot(results)
"""

"""
solver = solver.FemSolver(mesh)
u_sol = solver.solve(helmholtz, degree)

visualiser = vs.Visualiser()
plot(u_sol)
visualiser.plot_3d(u_sol, mesh)
"""