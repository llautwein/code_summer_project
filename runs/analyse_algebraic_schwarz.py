from HelperModules import visualiser, analyser
from dolfin import *
import ProblemDefinition as problem_def
import numpy as np

left_bottom_corner = (0, -0.25)
mid_intersection = 0.75
overall_height = 2
mesh_resolutions = [0.01]
polynomial_degrees = [1]
interface_widths = np.logspace(-1, -3, 10)

tol = 1e-6
max_iter = 1000
f_1 = Expression("(8*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2)
g_1 = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=6)
model_problem = problem_def.ModelProblem(f_1)

g_2 = Expression("1 + x[0]*x[0] + 2 * x[1]*x[1]", degree=2)
f_2 = Constant(-6.0)
poisson = problem_def.PoissonProblem(f_2)

results_path = "output_files/algebraic_schwarz_analysis_conforming.csv"
analyser = analyser.Analyser()
analyser.analyse_algebraic_schwarz_conforming(left_bottom_corner, overall_height, mid_intersection,
                                   polynomial_degrees, interface_widths, tol,
                                   max_iter, model_problem, g_1, model_problem, g_1, results_path)

