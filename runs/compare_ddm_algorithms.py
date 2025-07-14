from HelperModules import visualiser, analyser
from dolfin import *
import ProblemDefinition as problem_def
import numpy as np

mesh_resolutions = np.logspace(-0.5, -4, 50)
tol = 1e-6
max_iter = 100
f_1 = Expression("(8*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2)
g_1 = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=6)
model_problem = problem_def.ModelProblem(f_1)
results_path = "output_files/schwarz_method_comparison.csv"
analyser = analyser.Analyser()
analyser.compare_ddm_algorithms((0, 0), 1, 0.55, (0, 0.45), 1, 0.55,
                                mesh_resolutions, tol, max_iter, model_problem, g_1, model_problem, g_1, results_path)




