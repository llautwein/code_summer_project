import FemSolver
from dolfin import *
import CompositionMethod as cm
import InterfaceHandler as ih
from HelperModules import GeometryParser as gp
import time
import pandas as pd
import numpy as np
from scipy import stats
from Config import ConformingMeshAnalysisConfig, IndependentMeshAnalysisConfig, DDMComparisonConfig



class Analyser:
    """
    A collection of static methods to run and post-process numerical experiments.
    """

    @staticmethod
    def run_convergence_analysis(problem, g, polynomial_degrees, element_sizes, u_exact):
        """
        Verifies the convergence result for refining finite element sizes from book of Quarteroni.
        :param problem: The problem to solve.
        :param g: The boundary condition.
        :param polynomial_degrees: The polynomial degrees of the Lagrange basis functions.
        :param element_sizes: A list of sizes for the finite elements.
        :param u_exact: The analytical solution of the problem.
        :return: A dictionary containing the convergence results.
        """
        results = {}
        solver = FemSolver.FemSolver()
        for d in polynomial_degrees:
            errors_L2 = []
            errors_H1 = []

            for h in element_sizes:
                n = int(round(1 / h))
                print(f"Degree d={d}, step size h={h}, n={n}")
                mesh = UnitSquareMesh(n, n)
                V = FunctionSpace(mesh, "CG", d)
                bcs = DirichletBC(V, g, "on_boundary")
                u = TrialFunction(V)
                v = TestFunction(V)
                a_form = problem.a(u, v)
                L_form = problem.L(v)

                u_h = solver.solve(V, a_form, L_form, bcs)

                errors_L2.append(errornorm(u_exact, u_h, 'L2', mesh=mesh))
                errors_H1.append(errornorm(u_exact, u_h, 'H1', mesh=mesh))

            results[d] = {'h': element_sizes, 'L2': errors_L2, 'H1': errors_H1}
        return results

    @staticmethod
    def compare_ddm_algorithms(config: DDMComparisonConfig):
        results = []
        for h in config.mesh_resolutions:
            for delta in config.interface_widths:
                print(f"Running experiment for characteristic element length {h}")
                geo_parser = gp.GeometryParser()
                rec_upper, rec_lower = geo_parser.create_independent_meshes(config.left_bottom_corner,
                                                                            config.length, config.height,
                                                                            config.mid_intersection, delta, h,
                                                                            mesh_option=config.mesh_option,
                                                                            gmsh_parameters=config.gmsh_parameters)

                interface_handler = ih.OverlappingRectanglesInterfaceHandler(rec_upper, rec_lower)

                boundary_markers_1, boundary_markers_2 = interface_handler.mark_interface_boundaries(
                    config.mid_intersection - delta / 2, config.mid_intersection + delta / 2)
                V_1 = FunctionSpace(rec_upper, "CG", 1)
                V_2 = FunctionSpace(rec_lower, "CG", 1)
                total_dofs = V_1.dim() + V_2.dim()

                print("Alternating Schwarz method")
                schwarz_alternating = cm.SchwarzMethodAlternating(V_1, rec_upper, boundary_markers_1, config.problem_1, config.g_1,
                                                                  V_2, rec_lower, boundary_markers_2, config.problem_2, config.g_2)
                start_time_alt = time.perf_counter()
                u1, u2 = schwarz_alternating.solve(config.tol, config.max_iter)
                end_time_alt = time.perf_counter()
                results.append({
                    'Method': 'Alternating',
                    'Mesh Size (h)': h,
                    'Total DoFs': total_dofs,
                    'Time (s)': end_time_alt - start_time_alt,
                    'Iterations': schwarz_alternating.get_last_iteration_count()
                })

                print("Algebraic Schwarz method")
                start_setup_time_alg = time.perf_counter()
                schwarz_algebraic = cm.SchwarzMethodAlgebraic(V_1, rec_upper, boundary_markers_1, config.problem_1, config.g_1,
                                                              V_2, rec_lower, boundary_markers_2, config.problem_2, config.g_2)
                end_setup_time_alg = time.perf_counter()

                start_solve_time_alg = time.perf_counter()
                u1, u2 = schwarz_algebraic.solve(config.tol, config.max_iter)
                end_solve_time_alg = time.perf_counter()

                results.append({
                    'Method': 'Algebraic',
                    'Mesh Size (h)': h,
                    'Total DoFs': total_dofs,
                    'Time (s)': (end_setup_time_alg - start_setup_time_alg) + (end_solve_time_alg - start_solve_time_alg),
                    'Iterations': schwarz_algebraic.get_last_iteration_count()
                })

                print("Matrix free Schwarz method")
                start_setup_time_mf = time.perf_counter()
                schwarz_mf = cm.SchwarzMethodMatrixFree(V_1, rec_upper, boundary_markers_1, config.problem_1, config.g_1,
                                                        V_2, rec_lower, boundary_markers_2, config.problem_2, config.g_2)
                end_setup_time_mf = time.perf_counter()

                start_solve_time_mf = time.perf_counter()
                u1, u2 = schwarz_mf.solve(config.tol, config.max_iter)
                end_solve_time_mf = time.perf_counter()

                results.append({
                    'Method': 'Matrix-Free',
                    'Mesh Size (h)': h,
                    'Total DoFs': total_dofs,
                    'Time (s)': (end_setup_time_mf - start_setup_time_mf) + (end_solve_time_mf - start_solve_time_mf),
                    'Iterations': schwarz_mf.get_last_iteration_count()
                })

        df = pd.DataFrame(results)
        df.to_csv(config.results_path, index=False)

    def analyse_algebraic_schwarz_independent(self, config: IndependentMeshAnalysisConfig):
        """
        Investigates the performance in terms of iterations and run time of the algebraic Schwarz method
        for the given mesh sizes, polynomial degrees and overlap widths.
        In this case, the meshes are generated independent of each other using either built-in fenics
        functions or a gmsh approach.

        :param config: A dataclass object containing all experiment parameters.
        """
        results = []
        # Loop through every possible combination
        for h in config.mesh_resolutions:
            for d in config.polynomial_degrees:
                for delta in config.interface_widths:
                    print(f"Running the analysis for h={h}, d={d}, delta={delta}")
                    # Create the two independent rectangular meshes and define the function spaces
                    geo_parser = gp.GeometryParser()
                    rec_upper, rec_lower = geo_parser.create_independent_meshes(config.left_bottom_corner,
                                                                                config.length, config.height,
                                                                                config.mid_intersection, delta, h,
                                                                                mesh_option=config.mesh_option,
                                                                                gmsh_parameters=config.gmsh_parameters)
                    V_1 = FunctionSpace(rec_upper, "CG", d)
                    V_2 = FunctionSpace(rec_lower, "CG", d)
                    total_dofs = V_1.dim() + V_2.dim()
                    # Mark the physical and artificial part of the boundary
                    interface_handler = ih.OverlappingRectanglesInterfaceHandler(rec_upper, rec_lower)
                    boundary_markers_1, boundary_markers_2 = interface_handler.mark_interface_boundaries(
                        config.mid_intersection - delta / 2,
                        config.mid_intersection + delta / 2
                    )
                    # Set up the algorithm
                    start_setup_time_alg = time.perf_counter()
                    schwarz_algebraic = cm.SchwarzMethodAlgebraic(V_1, rec_upper, boundary_markers_1, config.problem_1,
                                                                  config.g_1,
                                                                  V_2, rec_lower, boundary_markers_2, config.problem_2,
                                                                  config.g_2, use_lu_decomposition=config.use_lu_solver)
                    end_setup_time_alg = time.perf_counter()

                    # Solve the problem using the algebraic Schwarz method
                    start_solve_time_alg = time.perf_counter()
                    u1, u2 = schwarz_algebraic.solve(config.tol, config.max_iter)
                    end_solve_time_alg = time.perf_counter()
                    """
                    # Uncomment to verify that the error is small
                    sol_analytic_rec_1 = interpolate(g_1, V_1)
                    sol_analytic_rec_2 = interpolate(g_1, V_2)
                    print(f"Error of u1: {errornorm(sol_analytic_rec_1, u1, 'L2', mesh=mesh_rectangle_1)}")
                    print(f"Error of u2: {errornorm(sol_analytic_rec_2, u2, 'L2', mesh=mesh_rectangle_2)}")
                    """
                    results.append({
                        'Mesh Size (h)': h,
                        'Polynomial Degree d': d,
                        'Interface Width': delta,
                        'Total DoFs': total_dofs,
                        'Time (s)': (end_setup_time_alg - start_setup_time_alg) + (
                                end_solve_time_alg - start_solve_time_alg),
                        'Iterations': schwarz_algebraic.get_last_iteration_count()
                    })

                # Post-processing of the results
                df = pd.DataFrame(results)
                # We expect: iterations ~ C*(1/delta)^m, thus add the x-axis for the fit
                df['1 / Interface Width'] = 1.0 / df['Interface Width']

                # Perform the fit and add annotated columns ('Fit Iterations', 'Fit Slope')
                df_final = self.fit_and_annotate_dataframe(
                    df,
                    x_col='1 / Interface Width',
                    y_col='Iterations',
                    group_by_col='Polynomial Degree d' # Group by degree within this subset
                )

                df_final.to_csv(config.results_path, index=False)
                print(f"\nAnalysis complete. Results saved to {config.results_path}")

    def analyse_algebraic_schwarz_conforming(self, config: ConformingMeshAnalysisConfig):
        """
        Investigates the performance in terms of iterations and run time of the algebraic Schwarz method
        for the given mesh sizes, polynomial degrees and overlap widths.
        In this case, the one base mesh of the overall domain is used and then divided into the subdomains.
        In the overlap region, there is one N_overlap layers of finite elements which both subdomains
        have in common.

        :param config: A dataclass object containing all experiment parameters.
        """
        results = []
        for d in config.polynomial_degrees:
            for delta in config.interface_widths:
                print(f"Running the analysis for d={d}, delta={delta}")
                geo_parser = gp.GeometryParser()
                rec_upper, rec_lower = geo_parser.create_conforming_meshes(config.left_bottom_corner,
                                                                           config.length, config.height,
                                                                           config.mid_intersection,
                                                                           delta, N_overlap=1,
                                                                           mesh_option=config.mesh_option,
                                                                           gmsh_parameters=config.gmsh_parameters)

                interface_handler = ih.OverlappingRectanglesInterfaceHandler(rec_upper, rec_lower)

                boundary_markers_1, boundary_markers_2 = interface_handler.mark_interface_boundaries(
                    config.mid_intersection - delta / 2, config.mid_intersection + delta / 2)
                V_1 = FunctionSpace(rec_upper, "CG", d)
                V_2 = FunctionSpace(rec_lower, "CG", d)
                total_dofs = V_1.dim() + V_2.dim()

                start_setup_time_alg = time.perf_counter()
                schwarz_algebraic = cm.SchwarzMethodAlgebraic(V_1, rec_upper, boundary_markers_1,
                                                              config.problem_1,
                                                              config.g_1,
                                                              V_2, rec_lower, boundary_markers_2,
                                                              config.problem_2,
                                                              config.g_2, use_lu_decomposition=config.use_lu_solver)
                end_setup_time_alg = time.perf_counter()

                start_solve_time_alg = time.perf_counter()
                u1, u2 = schwarz_algebraic.solve(config.tol, config.max_iter)
                end_solve_time_alg = time.perf_counter()


                # Uncomment to verify that the error is small
                sol_analytic_rec_1 = interpolate(config.g_1, V_1)
                sol_analytic_rec_2 = interpolate(config.g_1, V_2)
                print(f"Error of u1: {errornorm(sol_analytic_rec_1, u1, 'L2', mesh=rec_upper)}")
                print(f"Error of u2: {errornorm(sol_analytic_rec_2, u2, 'L2', mesh=rec_lower)}")


                results.append({
                    'Polynomial Degree d': d,
                    'Interface Width': delta,
                    'Total DoFs': total_dofs,
                    'Time (s)': (end_setup_time_alg - start_setup_time_alg) + (
                            end_solve_time_alg - start_solve_time_alg),
                    'Iterations': schwarz_algebraic.get_last_iteration_count()
                })

        df = pd.DataFrame(results)
        df['1 / Interface Width'] = 1.0 / df['Interface Width']

        df_final = self.fit_and_annotate_dataframe(
                df,
                x_col='1 / Interface Width',
                y_col='Iterations',
                group_by_col='Polynomial Degree d'
            )

        df_final.to_csv(config.results_path, index=False)
        print(f"\nAnalysis complete. Results saved to {config.results_path}")

    @staticmethod
    def fit_and_annotate_dataframe(df, x_col, y_col, group_by_col):
        """
        Fits a power law to grouped data and annotates the DataFrame with fit results.
        y = C * x^m  --> log(y) = log(C) + m*log(x)
        """
        df_fit = df.copy()
        df_fit['Fit ' + y_col] = np.nan
        df_fit['Fit Slope'] = np.nan

        for name, group in df_fit.groupby(group_by_col):
            if len(group) < 2:
                print(f"Skipping fit for group '{name}': not enough data points.")
                continue

            # Sort values for correct line plotting
            group = group.sort_values(by=x_col)

            # Use log-log data for linear regression to find the power law exponent
            log_x = np.log(group[x_col])
            log_y = np.log(group[y_col])

            slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)

            print(f"\nFit results for group: {group_by_col}={name}")
            print(f"  - Power Law Exponent (Slope m): {slope:.4f}")
            print(f"  - R-squared: {r_value ** 2:.4f}")

            # Calculate fitted y-values and store them
            y_fit = np.exp(intercept) * (group[x_col] ** slope)

            # Place results back into the original DataFrame's indices
            df_fit.loc[group.index, 'Fit ' + y_col] = y_fit
            df_fit.loc[group.index, 'Fit Slope'] = slope

        return df_fit
