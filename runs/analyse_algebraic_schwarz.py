from HelperModules import visualiser, analyser
from dolfin import *
import ProblemDefinition as problem_def
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Config import ConformingMeshAnalysisConfig, IndependentMeshAnalysisConfig, OffsetMeshAnalysisConfig

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
else:
    raise ValueError("Invalid mode")

visualiser = visualiser.Visualiser()
fixed_params = {"Interface Width": config.interface_widths}
compare_by = "Interface Width"
df = pd.read_csv(config.results_path)

sns.lineplot(df, x=df["Offset Percentage"], y=df["Iterations"], hue=compare_by)
plt.show()
"""
visualiser.analyse_algebraic_schwarz_plot(
    config.results_path,"Offset Percentage", "Iterations", fixed_params, compare_by
)"""

