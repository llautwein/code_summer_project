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
results_path_scalability = "output_files/algebraic_schwarz_analysis_scalability.csv"
# Mesh Size (h), Polynomial Degree d, Interface Width, Total DoFs, Time (s), Iterations

def filter_dataframe(df: pd.DataFrame, fixed_col: str, fixed_vals_idcs: list) -> pd.DataFrame:
    fixed_val = df[fixed_col].unique()[fixed_vals_idcs]
    mask = np.any([np.isclose(df[fixed_col], val) for val in fixed_val], axis=0)
    return df[mask]
"""
sns.set_theme(style="whitegrid")
sns.set_palette("pastel")
df = pd.read_csv(results_path_offset)
df_fixed_offset = filter_dataframe(df, "Offset Percentage", [0,3])
print(df_fixed_offset)
df_final = filter_dataframe(df_fixed_offset, "Polynomial Degree d", [0])
print(df_final)

sns.lineplot(df_final, x="Interface Width", y="Iterations", hue="Offset Percentage")
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
"""
fixed_params = {"Polynomial Degree d": [1]}
compare_by = "Polynomial Degree d"
visualiser.analyse_algebraic_schwarz_plot(results_path_independent,
                                    "Interface Width", "Iterations", fixed_params, compare_by)


"""
scenario = {"conforming": results_path_conforming, "independent": results_path_independent}
#visualiser.iterations_delta_scenarios_plot(scenario, "Interface Width", "Iterations", None)
visualiser.plot_parameter_study(
    results_path_scalability,
    "Total DoFs",
    "Iterations",
      "Polynomial Degree d",
    fixed_params={"Interface Width": [0.01]},
    plot_fit=False,
    x_log=True, y_log=True
)