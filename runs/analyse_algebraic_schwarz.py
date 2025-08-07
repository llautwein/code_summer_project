from HelperModules import visualiser, analyser
from dolfin import *
import ProblemDefinition as problem_def
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Config import (ConformingMeshAnalysisConfig, IndependentMeshAnalysisConfig,
                    OffsetMeshAnalysisConfig, MeshAnalysis3d)

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
offset_config = OffsetMeshAnalysisConfig(
    problem_1=model_problem, g_1=g_1,
    problem_2=model_problem, g_2=g_1
)
g_1_3d = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])", degree=6)
f_1_3d = Expression("(12*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])", degree=2)
model_problem_3d = problem_def.ModelProblem(f_1_3d)
config_3d = MeshAnalysis3d(
    problem_1=model_problem_3d, g_1=g_1_3d,
    problem_2=model_problem_3d, g_2=g_1_3d
)

mode = "offset"
analyser = analyser.Analyser()
if mode=="conforming":
    config = conforming_config
    analyser.analyse_algebraic_schwarz_conforming(config)
elif mode=="independent":
    config = independent_config
    analyser.analyse_algebraic_schwarz_independent(config)
elif mode=="offset":
    config = offset_config
    analyser.analyse_algebraic_schwarz_offset(config)
elif mode=="3d":
    config = config_3d
    analyser.analyse_algebraic_schwarz_3d(config)
else:
    raise ValueError("Invalid mode")

visualiser = visualiser.Visualiser()
fixed_params = {"Polynomial Degree d": config.polynomial_degrees}
compare_by = "Polynomial Degree d"

visualiser.analyse_algebraic_schwarz_plot(
    config.results_path,"Interface Width", "Iterations", fixed_params, compare_by
)
exit()
df = pd.read_csv(config.results_path)
sns.lineplot(df, x=df["Offset Percentage"], y=df["Iterations"], hue=compare_by)
plt.show()
