import ProblemDefinition as problem_def
from dolfin import *
import numpy as np
from HelperModules import visualiser, analyser

# Run that verifies the convergence of the finite element method.
# Shows relationship of the error, the polynomial degree and the size of the elements.

V = FunctionSpace(mesh, "CG", 1)
g = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=6)
f = Expression("(8*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2)
model_problem = problem_def.ModelProblem(f)
polynomial_degrees = [1, 2]
step_sizes = np.logspace(np.log10(np.sqrt(0.001)), np.log10(np.sqrt(0.1)), num=4)
analyser = analyser.Analyser()
results = analyser.run_convergence_analysis(model_problem, g, polynomial_degrees,
                                            step_sizes, g)
visualiser = visualiser.Visualiser()
visualiser.convergence_rates_plot(results)

