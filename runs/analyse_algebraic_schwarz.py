from HelperModules import visualiser, analyser
from dolfin import *
import ProblemDefinition as problem_def
from Config import ConformingMeshAnalysisConfig, IndependentMeshAnalysisConfig

f_1 = Expression("(8*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2)
g_1 = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=6)
model_problem = problem_def.ModelProblem(f_1)
conforming_config = ConformingMeshAnalysisConfig(
    problem_1=model_problem, g_1=g_1,
    problem_2=model_problem, g_2=g_1
)
independent_config = IndependentMeshAnalysisConfig(
    problem_1=model_problem, g_1=g_1,
    problem_2=model_problem, g_2=g_1
)

#config = conforming_config
config = independent_config
analyser = analyser.Analyser()
#analyser.analyse_algebraic_schwarz_conforming(config)
analyser.analyse_algebraic_schwarz_independent(config)

visualiser = visualiser.Visualiser()
fixed_params = {"Polynomial Degree d": [1]}
compare_by = "Polynomial Degree d"
visualiser.analyse_algebraic_schwarz_plot(
    config.results_path,"Interface Width", "Iterations", fixed_params, compare_by
)

