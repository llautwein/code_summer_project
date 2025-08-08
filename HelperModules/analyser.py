import FemSolver
from dolfin import *
import CompositionMethod as cm
import InterfaceHandler as ih
from HelperModules import GeometryParser as gp
import time
import pandas as pd
import numpy as np
from scipy import stats
from Config import (ConformingMeshAnalysisConfig, IndependentMeshAnalysisConfig,
                    OffsetMeshAnalysisConfig, DDMComparisonConfig, ScalabilityAnalysisConfig, MeshAnalysis3d)



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
                V_1 = FunctionSpace(rec_upper, "CG", 1)
                V_2 = FunctionSpace(rec_lower, "CG", 1)
                total_dofs = V_1.dim() + V_2.dim()
                interface_handler = ih.OverlappingRectanglesInterfaceHandler(rec_upper, rec_lower)

                boundary_markers_1, boundary_markers_2 = interface_handler.mark_interface_boundaries(
                    config.mid_intersection - delta / 2, config.mid_intersection + delta / 2)

                # Extract the number of DoFs of the interface problem
                dofs_1 = cm.CompositionMethod.get_dof_indices(V_1, boundary_markers_1, 1, 2)
                dofs_2 = cm.CompositionMethod.get_dof_indices(V_2, boundary_markers_2, 1, 2)
                interface_problem_size = len(dofs_1['interface']) + len(dofs_2['interface'])

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
                    'Interface DoFs': interface_problem_size,
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
                    'Interface DoFs': interface_problem_size,
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
                    'Interface DoFs': interface_problem_size,
                    'Time (s)': (end_setup_time_mf - start_setup_time_mf) + (end_solve_time_mf - start_solve_time_mf),
                    'Iterations': schwarz_mf.get_last_iteration_count()
                })

        df = pd.DataFrame(results)
        df.to_csv(config.results_path, index=False)

    def _run_single_simulation(self, config, mesh_params):
        """
        Private helper method to run one simulation for a given set of parameters.
        This contains the core logic that was duplicated in all your analyse_... methods.
        """
        # Unpack mesh parameters
        h = mesh_params.get('h', None)
        d = mesh_params.get('d')
        delta = mesh_params.get('delta')
        delta_1 = mesh_params.get('delta_1')
        delta_2_pctg = mesh_params.get('delta_2_pctg')

        # Mesh setup
        geo_parser = gp.GeometryParser()

        # Determine which meshing function to call based on the config type
        # Compute boundary markers
        if isinstance(config, IndependentMeshAnalysisConfig):
            mesh1, mesh2 = geo_parser.create_independent_meshes(
                config.left_bottom_corner, config.length, config.height,
                config.mid_intersection, delta, h,
                mesh_option=config.mesh_option, gmsh_parameters=config.gmsh_parameters
            )
            interface_handler = ih.OverlappingRectanglesInterfaceHandler(mesh1, mesh2)
            y_upper = config.mid_intersection - (delta or delta_1) / 2
            y_lower = config.mid_intersection + (delta or delta_1) / 2
            bm1, bm2 = interface_handler.mark_interface_boundaries(y_upper, y_lower)
        elif isinstance(config, OffsetMeshAnalysisConfig):
            mesh1, mesh2 = geo_parser.create_offset_meshes(
                config.left_bottom_corner, config.length, config.height,
                config.mid_intersection, delta_1, delta_2_pctg, h,
                gmsh_parameters=config.gmsh_parameters
            )
            interface_handler = ih.InterfaceHandler(mesh1, mesh2)
            bm1, bm2 = interface_handler.mark_interface_boundaries(2)

        elif isinstance(config, ConformingMeshAnalysisConfig):
            mesh1, mesh2 = geo_parser.create_conforming_meshes(
                config.left_bottom_corner, config.length, config.height,
                config.mid_intersection, delta, N_overlap=1,
                mesh_option=config.mesh_option, gmsh_parameters=config.gmsh_parameters
            )
            interface_handler = ih.OverlappingRectanglesInterfaceHandler(mesh1, mesh2)
            y_upper = config.mid_intersection - (delta or delta_1) / 2
            y_lower = config.mid_intersection + (delta or delta_1) / 2
            bm1, bm2 = interface_handler.mark_interface_boundaries(y_upper, y_lower)
        elif isinstance(config, ScalabilityAnalysisConfig):
            mesh1, mesh2 = geo_parser.create_independent_meshes(
                config.left_bottom_corner, config.length, config.height,
                config.mid_intersection, delta, h,
                mesh_option=config.mesh_option, gmsh_parameters=config.gmsh_parameters
            )
            interface_handler = ih.OverlappingRectanglesInterfaceHandler(mesh1, mesh2)
            y_upper = config.mid_intersection - (delta or delta_1) / 2
            y_lower = config.mid_intersection + (delta or delta_1) / 2
            bm1, bm2 = interface_handler.mark_interface_boundaries(y_upper, y_lower)
        elif isinstance(config, MeshAnalysis3d):
            mesh1, mesh2 = geo_parser.create_3d_meshes(
                config.left_bottom_corner, config.length, config.width, config.height,
                config.mid_intersection, delta, h,
                gmsh_parameters=config.gmsh_parameters
            )
            interface_handler = ih.OverlappingRectanglesInterfaceHandler(mesh1, mesh2)
            z_upper = config.mid_intersection - delta / 2
            z_lower = config.mid_intersection + delta / 2
            bm1, bm2 = interface_handler.mark_interface_boundaries_3d(z_upper, z_lower)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        V_1 = FunctionSpace(mesh1, "CG", d)
        V_2 = FunctionSpace(mesh2, "CG", d)

        # Interface Size Calculation
        dofs_1 = cm.CompositionMethod.get_dof_indices(V_1, bm1, 1, 2)
        dofs_2 = cm.CompositionMethod.get_dof_indices(V_2, bm2, 1, 2)
        interface_problem_size = len(dofs_1['interface']) + len(dofs_2['interface'])

        # Solver Execution
        start_time = time.perf_counter()
        schwarz_algebraic = cm.SchwarzMethodAlgebraic(
            V_1, mesh1, bm1, config.problem_1, config.g_1,
            V_2, mesh2, bm2, config.problem_2, config.g_2,
            use_lu_decomposition=config.use_lu_solver
        )
        u1, u2 = schwarz_algebraic.solve(config.tol, config.max_iter)
        end_time = time.perf_counter()

        # Return Results Dictionary
        return {
            'Mesh Size (h)': h,
            'Polynomial Degree d': d,
            'Interface Width': delta or delta_1,
            'Offset Percentage': delta_2_pctg,
            'Total DoFs': V_1.dim() + V_2.dim(),
            'Interface DoFs': interface_problem_size,
            'Total Time (s)': end_time - start_time,
            'Iterations': schwarz_algebraic.get_last_iteration_count()
        }

    def analyse_independent_meshes(self, config: IndependentMeshAnalysisConfig):
        results = []
        for h in config.mesh_resolutions:
            for d in config.polynomial_degrees:
                for delta in config.interface_widths:
                    print(f"Running Independent: h={h}, d={d}, delta={delta:.4f}")
                    params = {'h': h, 'd': d, 'delta': delta}
                    results.append(self._run_single_simulation(config, params))
        self.post_process(results, ['Mesh Size (h)', 'Polynomial Degree d'], config.results_path)

    def analyse_conforming_meshes(self, config: ConformingMeshAnalysisConfig):
        results = []
        for d in config.polynomial_degrees:
            for delta in config.interface_widths:
                print(f"Running Conforming: d={d}, delta={delta:.4f}")
                params = {'d': d, 'delta': delta}
                results.append(self._run_single_simulation(config, params))
        self.post_process(results, ['Polynomial Degree d'], config.results_path)

    def analyse_offset_meshes(self, config: OffsetMeshAnalysisConfig):
        results = []
        for h in config.mesh_resolutions:
            for d in config.polynomial_degrees:
                for delta_1 in config.interface_widths:
                    for delta_2_pctg in config.offset_pctg:
                        print(f"Running Offset: h={h}, d={d}, delta_1={delta_1:.4f}, offset={delta_2_pctg:.2f}")
                        params = {'h': h, 'd': d, 'delta_1': delta_1, 'delta_2_pctg': delta_2_pctg}
                        results.append(self._run_single_simulation(config, params))
        self.post_process(results, ['Mesh Size (h)', 'Polynomial Degree d', 'Offset Percentage'], config.results_path)

    def analyse_scalability(self, config: ScalabilityAnalysisConfig):
        results = []
        for h in config.mesh_resolutions:
            for d in config.polynomial_degrees:
                for delta in config.interface_widths:
                    print(f"Running Scalability: h={h}, d={d}, delta={delta:.4f}")
                    params = {'h': h, 'd': d, 'delta': delta}
                    results.append(self._run_single_simulation(config, params))
        df = pd.DataFrame(results)
        df.to_csv(config.results_path, index=False)
        print(f"\nScalability analysis complete. Results saved to {config.results_path}")

    def analyse_3d_meshes(self, config: MeshAnalysis3d):
        results = []
        for h in config.mesh_resolutions:
            for d in config.polynomial_degrees:
                for delta in config.interface_widths:
                    print(f"Running 3D: h={h}, d={d}, delta={delta:.4f}")
                    params = {'h': h, 'd': d, 'delta': delta}
                    results.append(self._run_single_simulation(config, params))
        self.post_process(results, ['Mesh Size (h)', 'Polynomial Degree d'], config.results_path)

    def post_process(self, results, grouping_cols, results_path):
        """
        Post-processing of the results. Creates a pandas DataFrame of the results dict, groups it by the
        specifies columns, and fits the power law in dependence of the interface width for every unique combination.
        :param results: The dict with the results.
        :param grouping_cols: The columns to group by.
        :param results_path: Path where to save the results.
        :return:
        """
        df = pd.DataFrame(results)
        print(df)
        df['1 / Interface Width'] = 1.0 / df['Interface Width']

        df_final = df.groupby(grouping_cols, group_keys=False).apply(
            self.fit_power_law,
            x_col='1 / Interface Width',
            y_col='Iterations'
        )

        df_final.to_csv(results_path, index=False)
        print(f"\nAnalysis complete. Results saved to {results_path}")

    @staticmethod
    def fit_power_law(group, x_col, y_col):
        """
        Fits a power law y = C * x^m to a given group of data (a DataFrame subset).
        This function is designed to be used with pandas' groupby().apply().
        It adds new columns to the group containing the fit results.
        """
        # Make a copy to avoid modifying the original slice
        df_fit = group.copy()

        # A fit requires at least 2 data points.
        if len(df_fit) < 2:
            df_fit['Fit ' + y_col] = np.nan
            df_fit['Fit Slope'] = np.nan
            df_fit['Fit R2'] = np.nan
            return df_fit

        # Sort values to ensure the fitted line plots correctly
        df_fit = df_fit.sort_values(by=x_col)

        # Perform log-log linear regression to find the exponent 'm' (the slope)
        log_x = np.log(df_fit[x_col].values + 1e-9)
        log_y = np.log(df_fit[y_col].values + 1e-9)
        slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)

        # Calculate the fitted y-values based on the power law
        y_fit = np.exp(intercept) * (df_fit[x_col].values ** slope)

        # Add the new analysis columns to the DataFrame group
        df_fit['Fit ' + y_col] = y_fit
        df_fit['Fit Slope'] = slope
        df_fit['Fit R2'] = r_value ** 2

        return df_fit
