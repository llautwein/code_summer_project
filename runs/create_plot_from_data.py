import matplotlib.pyplot as plt

from HelperModules import visualiser
import numpy as np
import pandas as pd
import seaborn as sns

visualiser = visualiser.Visualiser()
"""
# Data from the method comparison results
method_comparison_results_path = "output_files/schwarz_method_comparison.csv"
visualiser.compare_ddm_methods_plot(method_comparison_results_path)
"""

# Data from the analysis of the algebraic Schwarz method
results_path_independent = "output_files/algebraic_schwarz_analysis_independent.csv"
results_path_conforming = "output_files/algebraic_schwarz_analysis_conforming.csv"
results_path_offset = "output_files/algebraic_schwarz_analysis_offset.csv"
results_path_3d = "output_files/algebraic_schwarz_analysis_3d.csv"
# Mesh Size (h), Polynomial Degree d, Interface Width, Total DoFs, Time (s), Iterations

sns.set_theme(style="whitegrid")
sns.set_palette("pastel")
df = pd.read_csv(results_path_offset)
fixed_offset = df["Offset Percentage"].unique()[[0, 4, 9]]
mask_offset = np.any([np.isclose(df["Offset Percentage"], val) for val in fixed_offset], axis=0)
df_fixed_offset = df[mask_offset]

sns.lineplot(df_fixed_offset, x="Interface Width", y="Iterations", hue="Offset Percentage")
plt.xscale("log")
plt.yscale("log")
plt.show()
###################
fixed_delta = df["Interface Width"].unique()[[0, 6, 11]]

mask_delta = np.any([np.isclose(df["Interface Width"], val) for val in fixed_delta], axis=0)
df_fixed_delta = df[mask_delta]
df_fixed_delta = df_fixed_delta.copy()
df_fixed_delta["Interface Width"] = df_fixed_delta["Interface Width"].round(4)

sns.lineplot(data=df_fixed_delta, x="Offset Percentage", y="Iterations", hue="Interface Width")
plt.show()

"""
fixed_params = {"Polynomial Degree d": [1]}
compare_by = "Polynomial Degree d"
visualiser.analyse_algebraic_schwarz_plot(results_path_independent,
                                    "Interface Width", "Iterations", fixed_params, compare_by)
"""
#visualiser.iterations_delta_scenarios_plot(results_path_conforming, results_path_independent)