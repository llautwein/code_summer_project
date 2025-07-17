import FemSolver
from dolfin import *
import CompositionMethod as cm
import InterfaceHandler as ih
from HelperModules import GeometryParser as gp
import time
import pandas as pd
import numpy as np
from scipy import stats


class Analyser:
    """
    Verifies the convergence results for different stepsizes and polynomial degrees.
    """

    @staticmethod
    def run_convergence_analysis(problem, u0, polynomial_degrees, step_sizes, u_exact):
        results = {}
        solver = FemSolver.FemSolver()
        for d in polynomial_degrees:
            errors_L2 = []
            errors_H1 = []

            for h in step_sizes:
                n = int(round(1 / h))
                print(f"Degree d={d}, step size h={h}, n={n}")
                mesh = UnitSquareMesh(n, n)
                V = FunctionSpace(mesh, "CG", d)
                bcs = DirichletBC(V, u0, "on_boundary")
                u = TrialFunction(V)
                v = TestFunction(V)
                a_form = problem.a(u, v)
                L_form = problem.L(v)

                u_h = solver.solve(V, a_form, L_form, bcs)

                errors_L2.append(errornorm(u_exact, u_h, 'L2', mesh=mesh))
                errors_H1.append(errornorm(u_exact, u_h, 'H1', mesh=mesh))

            results[d] = {'h': step_sizes, 'L2': errors_L2, 'H1': errors_H1}
        return results

    @staticmethod
    def compare_ddm_algorithms(left_bottom_corner_1, length_1, height_1, left_bottom_corner_2, length_2, height_2,
                               mesh_resolutions, tol, max_iter, problem_1, g_1, problem_2, g_2, results_path):
        """
        Compares the different implementations of Schwarz's algorithm for two overlapping rectangular meshes.
        The parameter differed is the mesh resolution, i.e. the number of dofs, the analysed results involve
        the run time and the number of iterations each algorithm takes.
        :param left_bottom_corner_1: Left bottom corner of the lower rectangle.
        :param length_1: Length of lower rectangle
        :param height_1: Height of lower rectangle
        :param left_bottom_corner_2: Left bottom corner of upper rectangle
        :param length_2: Length of upper rectangle
        :param height_2: Height of upper rectangle
        :param mesh_resolutions: List of characteristic lengths of mesh elements
        :param tol: Tolerance of the algorithms
        :param max_iter: Maximum iteration of the algorithms
        :param problem_1: Problem formulation on subdomain 1
        :param g_1: True boundary condition on subdomain 1
        :param problem_2: Problem formulation on subdomain 2
        :param g_2: True boundary condition on subdomain 2
        :param results_path: Path to the results file.
        """
        results = []
        for h in mesh_resolutions:
            print(f"Running experiment for characteristic element length {h}")
            # Lower rectangle:
            # The point p0 is the left bottom corner, p1 the right top corner of the lower rectangle.
            p0_lower = Point(left_bottom_corner_1[0], left_bottom_corner_1[1])
            p1_lower = Point(left_bottom_corner_1[0] + length_1, left_bottom_corner_1[1] + height_1)

            # Upper rectangle (same principle):
            p0_upper = Point(left_bottom_corner_2[0], left_bottom_corner_2[1])
            p1_upper = Point(left_bottom_corner_2[0] + length_2, left_bottom_corner_2[1] + height_2)

            nx_upper = max(1, int(round(length_1 / h)))
            ny_upper = max(1, int(round(height_1 / h)))

            nx_lower = max(1, int(round(length_2 / h)))
            ny_lower = max(1, int(round(height_2 / h)))

            # 3. Create meshes directly in memory using FEniCS
            mesh_rectangle_1 = RectangleMesh(p0_upper, p1_upper, nx_upper, ny_upper)
            mesh_rectangle_2 = RectangleMesh(p0_lower, p1_lower, nx_lower, ny_lower)

            interface_handler = ih.OverlappingRectanglesInterfaceHandler(mesh_rectangle_1, mesh_rectangle_2)

            boundary_markers_1, boundary_markers_2 = interface_handler.mark_interface_boundaries(
                left_bottom_corner_2[1], height_1)

            V_1 = FunctionSpace(mesh_rectangle_1, "CG", 1)
            V_2 = FunctionSpace(mesh_rectangle_2, "CG", 1)
            total_dofs = V_1.dim() + V_2.dim()

            print("Alternating Schwarz method")
            schwarz_alternating = cm.SchwarzMethodAlternating(V_1, mesh_rectangle_1, boundary_markers_1, problem_1, g_1,
                                                              V_2, mesh_rectangle_2, boundary_markers_2, problem_2, g_2)
            start_time_alt = time.perf_counter()
            u1, u2 = schwarz_alternating.solve(tol, max_iter)
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
            schwarz_algebraic = cm.SchwarzMethodAlgebraic(V_1, mesh_rectangle_1, boundary_markers_1, problem_1, g_1,
                                                          V_2, mesh_rectangle_2, boundary_markers_2, problem_2, g_2)
            end_setup_time_alg = time.perf_counter()

            start_solve_time_alg = time.perf_counter()
            u1, u2 = schwarz_algebraic.solve(tol, max_iter)
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
            schwarz_mf = cm.SchwarzMethodMatrixFree(V_1, mesh_rectangle_1, boundary_markers_1, problem_1, g_1,
                                                    V_2, mesh_rectangle_2, boundary_markers_2, problem_2, g_2)
            end_setup_time_mf = time.perf_counter()

            start_solve_time_mf = time.perf_counter()
            u1, u2 = schwarz_mf.solve(tol, max_iter)
            end_solve_time_mf = time.perf_counter()

            results.append({
                'Method': 'Matrix-Free',
                'Mesh Size (h)': h,
                'Total DoFs': total_dofs,
                'Time (s)': (end_setup_time_mf - start_setup_time_mf) + (end_solve_time_mf - start_solve_time_mf),
                'Iterations': schwarz_mf.get_last_iteration_count()
            })

        df = pd.DataFrame(results)
        df.to_csv(results_path, index=False)

    def analyse_algebraic_schwarz_independent(self, left_bottom_corner_lower, overall_height, mid_intersection,
                                              mesh_resolutions, polynomial_degrees, interface_widths, tol,
                                              max_iter, problem_1, g_1, problem_2, g_2, results_path):
        length = 1
        results = []
        for h in mesh_resolutions:
            for d in polynomial_degrees:
                for delta in interface_widths:
                    print(f"Running the analysis for h={h}, d={d}, delta={delta}")
                    geo_parser = gp.GeometryParser()
                    mesh_rectangle_1, mesh_rectangle_2 = geo_parser.create_independent_meshes(left_bottom_corner_lower,
                                                                                              mid_intersection,
                                                                                              length, overall_height,
                                                                                              delta, h)

                    interface_handler = ih.OverlappingRectanglesInterfaceHandler(mesh_rectangle_1, mesh_rectangle_2)

                    boundary_markers_1, boundary_markers_2 = interface_handler.mark_interface_boundaries(
                        mid_intersection - delta / 2, mid_intersection + delta / 2)
                    V_1 = FunctionSpace(mesh_rectangle_1, "CG", d)
                    V_2 = FunctionSpace(mesh_rectangle_2, "CG", d)
                    total_dofs = V_1.dim() + V_2.dim()

                    start_setup_time_alg = time.perf_counter()
                    schwarz_algebraic = cm.SchwarzMethodAlgebraic(V_1, mesh_rectangle_1, boundary_markers_1, problem_1,
                                                                  g_1,
                                                                  V_2, mesh_rectangle_2, boundary_markers_2, problem_2,
                                                                  g_2)
                    end_setup_time_alg = time.perf_counter()

                    start_solve_time_alg = time.perf_counter()
                    u1, u2 = schwarz_algebraic.solve(tol, max_iter)
                    end_solve_time_alg = time.perf_counter()
                    sol_analytic_rec_1 = interpolate(g_1, V_1)
                    sol_analytic_rec_2 = interpolate(g_1, V_2)
                    print(f"Error of u1: {errornorm(sol_analytic_rec_1, u1, 'L2', mesh=mesh_rectangle_1)}")
                    print(f"Error of u2: {errornorm(sol_analytic_rec_2, u2, 'L2', mesh=mesh_rectangle_2)}")

                    results.append({
                        'Mesh Size (h)': h,
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

                df_final.to_csv(results_path, index=False)
                print(f"\nAnalysis complete. Results saved to {results_path}")

    def analyse_algebraic_schwarz_conforming(self, left_bottom_corner_lower, overall_height, mid_intersection,
                                             polynomial_degrees, interface_widths, tol,
                                             max_iter, problem_1, g_1, problem_2, g_2, results_path):
        length = 1
        results = []
        for d in polynomial_degrees:
            for delta in interface_widths:
                print(f"Running the analysis for d={d}, delta={delta}")
                geo_parser = gp.GeometryParser()
                mesh_rectangle_1, mesh_rectangle_2 = geo_parser.create_conforming_meshes(left_bottom_corner_lower,
                                                                                         length, overall_height,
                                                                                         mid_intersection,
                                                                                         delta, N_overlap=1)

                interface_handler = ih.OverlappingRectanglesInterfaceHandler(mesh_rectangle_1, mesh_rectangle_2)

                boundary_markers_1, boundary_markers_2 = interface_handler.mark_interface_boundaries(
                    mid_intersection - delta / 2, mid_intersection + delta / 2)
                V_1 = FunctionSpace(mesh_rectangle_1, "CG", d)
                V_2 = FunctionSpace(mesh_rectangle_2, "CG", d)
                total_dofs = V_1.dim() + V_2.dim()

                start_setup_time_alg = time.perf_counter()
                schwarz_algebraic = cm.SchwarzMethodAlgebraic(V_1, mesh_rectangle_1, boundary_markers_1, problem_1,
                                                              g_1,
                                                              V_2, mesh_rectangle_2, boundary_markers_2, problem_2,
                                                              g_2, False)
                end_setup_time_alg = time.perf_counter()

                start_solve_time_alg = time.perf_counter()
                u1, u2 = schwarz_algebraic.solve(tol, max_iter)
                end_solve_time_alg = time.perf_counter()

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

            df_final.to_csv(results_path, index=False)
            print(f"\nAnalysis complete. Results saved to {results_path}")

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
