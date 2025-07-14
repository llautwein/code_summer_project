import FemSolver
from dolfin import *
import CompositionMethod as cm
import InterfaceHandler as ih
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
                n = int(round(1/h))
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

            boundary_markers_1, boundary_markers_2 = interface_handler.mark_interface_boundaries(left_bottom_corner_2[1], height_1)

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


    @staticmethod
    def analyse_algebraic_schwarz(mesh_resolutions, polynomial_degrees, interface_widths, tol,
                                  max_iter, problem_1, g_1, problem_2, g_2, results_path):
        left_bottom_corner_lower = (0, 0)
        length = 1
        mid_intersection = 0.5
        results = []
        for h in mesh_resolutions:
            for d in polynomial_degrees:
                for delta in interface_widths:
                    print(f"Running the analysis for h={h}, d={d}, delta={delta}")
                    # Define the geometries the rectangles.
                    height = mid_intersection + delta / 2
                    # Lower rectangle:
                    # The point p0 is the left bottom corner, p1 the right top corner of the lower rectangle.
                    p0_lower = Point(left_bottom_corner_lower[0], left_bottom_corner_lower[1])
                    p1_lower = Point(left_bottom_corner_lower[0] + length, height)

                    # Upper rectangle (same principle):
                    p0_upper = Point(left_bottom_corner_lower[0], mid_intersection - delta / 2)
                    p1_upper = Point(left_bottom_corner_lower[0] + length, 1)

                    nx = max(1, int(round(length / h)))
                    ny = max(1, int(round(height / h)))

                    # Create meshes directly using FEniCS
                    mesh_rectangle_1 = RectangleMesh(p0_upper, p1_upper, nx, ny)
                    mesh_rectangle_2 = RectangleMesh(p0_lower, p1_lower, nx, ny)

                    interface_handler = ih.OverlappingRectanglesInterfaceHandler(mesh_rectangle_1, mesh_rectangle_2)

                    boundary_markers_1, boundary_markers_2 = interface_handler.mark_interface_boundaries(mid_intersection - delta/2, height)
                    V_1 = FunctionSpace(mesh_rectangle_1, "CG", d)
                    V_2 = FunctionSpace(mesh_rectangle_2, "CG", d)
                    total_dofs = V_1.dim() + V_2.dim()

                    start_setup_time_alg = time.perf_counter()
                    schwarz_algebraic = cm.SchwarzMethodAlgebraic(V_1, mesh_rectangle_1, boundary_markers_1, problem_1, g_1,
                                                                  V_2, mesh_rectangle_2, boundary_markers_2, problem_2, g_2)
                    end_setup_time_alg = time.perf_counter()

                    start_solve_time_alg = time.perf_counter()
                    u1, u2 = schwarz_algebraic.solve(tol, max_iter)
                    end_solve_time_alg = time.perf_counter()

                    results.append({
                        'Mesh Size (h)': h,
                        'Polynomial Degree': d,
                        'Interface Width': delta,
                        'Total DoFs': total_dofs,
                        'Time (s)': (end_setup_time_alg - start_setup_time_alg) + (
                                    end_solve_time_alg - start_solve_time_alg),
                        'Iterations': schwarz_algebraic.get_last_iteration_count()
                    })

                df = pd.DataFrame(results)
                df.to_csv(results_path, index=False)

                df['1 / Interface Width'] = 1.0 / (df['Interface Width'] + 1e-12)

                grouping_params = ['Mesh Size (h)', 'Polynomial Degree']

                for name, group in df.groupby(grouping_params):

                    h_val, d_val = name
                    print(f"\nAnalyzing group: h = {h_val:.4f}, degree = {d_val}")

                    # Ensure there are enough points to fit a line
                    if len(group) < 2:
                        print("  Not enough data points to perform a fit.")
                        continue

                    # Extract the relevant columns for fitting
                    # We sort by the x-axis value to ensure the data is ordered for plotting later
                    group = group.sort_values(by='Interface Width')
                    x_data = group['1 / Interface Width']
                    y_data = group['Iterations']

                    # Perform the linear regression on the log-transformed data
                    # This is equivalent to fitting a power law to the original data
                    log_x = np.log(x_data)
                    log_y = np.log(y_data)

                    # Use scipy.stats.linregress to get the slope and other stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

                    print("Parameters of the fit C*delta^(-m)")
                    print(f"  - Fit Slope m: {slope:.4f}")
                    print(f"  - R-squared: {r_value ** 2:.4f}")

