from HelperModules import visualiser
import numpy as np

visualiser = visualiser.Visualiser()
"""
# Data from the method comparison results
method_comparison_results_path = "output_files/schwarz_method_comparison.csv"
visualiser.compare_ddm_methods_plot(method_comparison_results_path)
"""

# Data from the analysis of the algebraic Schwarz method
algebraic_schwarz_analysis_results_path = "output_files/algebraic_schwarz_analysis_conforming.csv"
# Mesh Size (h), Polynomial Degree d, Interface Width, Total DoFs, Time (s), Iterations

fixed_params = {"Polynomial Degree d": [1]}
compare_by = "Polynomial Degree d"
visualiser.analyse_algebraic_schwarz_plot(algebraic_schwarz_analysis_results_path,
                                    "Interface Width", "Iterations", fixed_params, compare_by)
