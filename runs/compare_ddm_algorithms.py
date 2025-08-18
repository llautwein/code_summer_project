from HelperModules import visualiser, analyser
from dolfin import *
import ProblemDefinition as problem_def
from Config import DDMComparisonConfig

f_1 = Expression("(8*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2)
g_1 = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=6)
model_problem = problem_def.ModelProblem(f_1)
config = DDMComparisonConfig(
    problem_1=model_problem, g_1=g_1,
    problem_2=model_problem, g_2=g_1
)

analyser = analyser.Analyser()
analyser.compare_ddm_algorithms(config)
visualiser = visualiser.Visualiser()
visualiser.compare_ddm_methods_plot(results_path=config.results_path)


