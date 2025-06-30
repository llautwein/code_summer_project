import FemSolver as fem_solver
from dolfin import *
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator


class CompositionMethod:
    def __init__(self, V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                 V_2, mesh_2, boundary_markers_2, problem_def_2, g_2):
        self.mesh_1 = mesh_1
        self.boundary_markers_1 = boundary_markers_1
        self.problem_def_1 = problem_def_1
        self.V_1 = V_1
        self.g_1 = g_1  # original boundary condition of problem 1
        self.mesh_2 = mesh_2
        self.boundary_markers_2 = boundary_markers_2
        self.problem_def_2 = problem_def_2
        self.V_2 = V_2
        self.g_2 = g_2  # original boundary condition of problem 2
        self.fem_solver = fem_solver.FemSolver()

    def solve(self, tol, max_iter):
        pass


class SchwarzMethod_alternating(CompositionMethod):
    def __init__(self, V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                 V_2, mesh_2, boundary_markers_2, problem_def_2, g_2):
        super().__init__(V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                         V_2, mesh_2, boundary_markers_2, problem_def_2, g_2)

    def create_bcs(self, V, boundary_markers, bcs_definitions):
        bcs = []
        for value_expr, target_id in bcs_definitions:
            bcs.append(DirichletBC(V, value_expr, boundary_markers, target_id))
        return bcs

    def solve(self, tol, max_iter=10):
        error_1 = 0
        error_2 = 0
        w_1 = Function(self.V_1)
        w_2 = Function(self.V_2)
        u1_prev = Function(self.V_1)
        u2_prev = Function(self.V_2)
        for k in range(max_iter):
            if error_1 < tol and error_2 < tol and k > 0:
                print(f"Tolerance {error_1} reached after {k} iteration.")
                break

            u1 = Function(self.V_1)
            bcs_1 = self.create_bcs(self.V_1, self.boundary_markers_1,
                                    [(self.g_1, 1), (w_1, 2)])
            helper_u_1 = TrialFunction(self.V_1)
            helper_v_1 = TestFunction(self.V_1)
            u1 = self.fem_solver.solve(self.V_1, self.problem_def_1.a(helper_u_1, helper_v_1),
                                       self.problem_def_1.L(helper_v_1), bcs_1)

            u1.set_allow_extrapolation(True)
            w_2.interpolate(u1)

            u2 = Function(self.V_2)
            bcs_2 = self.create_bcs(self.V_2, self.boundary_markers_2,
                                    [(self.g_2, 1), (w_2, 2)])
            helper_u_2 = TrialFunction(self.V_2)
            helper_v_2 = TestFunction(self.V_2)
            u2 = self.fem_solver.solve(self.V_2, self.problem_def_2.a(helper_u_2, helper_v_2),
                                       self.problem_def_2.L(helper_v_2), bcs_2)
            u2.set_allow_extrapolation(True)
            w_1.interpolate(u2)
            error_1 = errornorm(u1, u1_prev, "l2")
            error_2 = errornorm(u2, u2_prev, "l2")
            u1_prev = u1.copy(deepcopy=True)
            u2_prev = u2.copy(deepcopy=True)
        return u1, u2


class SchwarzMethod_matrix(CompositionMethod):
    def __init__(self, V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                 V_2, mesh_2, boundary_markers_2, problem_def_2, g_2):
        super().__init__(V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                         V_2, mesh_2, boundary_markers_2, problem_def_2, g_2)

        # pre computations for domain 1
        self.dofs_1 = self.get_dof_indices(V_1, boundary_markers_1, 1, 2)
        self.A_1, self.f_1 = self.assemble_system(V_1, problem_def_1)

        self.A_1_interior = self.A_1[np.ix_(self.dofs_1["interior"], self.dofs_1["interior"])]
        self.A_1_gamma1 = self.A_1[np.ix_(self.dofs_1["interior"], self.dofs_1["interface"])]
        self.A_1_phys1 = self.A_1[np.ix_(self.dofs_1["interior"], self.dofs_1["physical"])]
        print(
            f"A_1_interior: {self.A_1_interior.shape}, A_1_gamma1: {self.A_1_gamma1.shape}, A_1_phys1: {self.A_1_phys1.shape}")

        self.g_1_vec = self.get_bc_values(g_1, V_1, self.dofs_1["physical"])
        print(f"g_1: {self.g_1_vec.shape}")

        self.f_bar_1 = self.f_1[self.dofs_1["interior"]] - self.A_1_phys1 @ self.g_1_vec
        print(f"f_bar_1: {self.f_bar_1.shape}")

        # pre computations for domain 2
        self.dofs_2 = self.get_dof_indices(V_2, boundary_markers_2, 1, 2)
        self.A_2, self.f_2 = self.assemble_system(V_2, problem_def_2)

        self.A_2_interior = self.A_2[np.ix_(self.dofs_2["interior"], self.dofs_2["interior"])]
        self.A_2_gamma2 = self.A_2[np.ix_(self.dofs_2["interior"], self.dofs_2["interface"])]
        self.A_2_phys2 = self.A_2[np.ix_(self.dofs_2["interior"], self.dofs_2["physical"])]
        print(
            f"A_2_interior: {self.A_2_interior.shape}, A_2_gamma2: {self.A_2_gamma2.shape}, A_2_phys2: {self.A_2_phys2.shape}")

        self.g_2_vec = self.get_bc_values(g_2, V_2, self.dofs_2["physical"])
        print(f"g_2: {self.g_2_vec.shape}")

        self.f_bar_2 = self.f_2[self.dofs_2["interior"]] - self.A_2_phys2 @ self.g_2_vec
        print(f"f_bar_2: {self.f_bar_2.shape}")

    @staticmethod
    def get_dof_indices(V, boundary_markers, physical_marker, interface_marker):
        """
        Function that returns the indices of dofs lying in interior, on the interface boundary, and
        the physical/true boundary
        :param V: the function space containing the dofs
        :param boundary_markers: mesh function which contains the boundary markers
        :param physical_marker: ID of the physical boundary
        :param interface_marker: ID of the interface boundary
        :return: dict with the needed indices
        """
        bc_interface = DirichletBC(V, Constant(0.0), boundary_markers, interface_marker)
        interface_dofs = np.array(list(bc_interface.get_boundary_values().keys()), dtype=np.int32)

        bc_physical = DirichletBC(V, Constant(0.0), boundary_markers, physical_marker)
        physical_dofs = np.array(list(bc_physical.get_boundary_values().keys()), dtype=np.int32)
        physical_dofs = np.setdiff1d(physical_dofs, interface_dofs).astype(np.int32)
        all_dofs = np.arange(V.dim())

        boundary_dofs = np.union1d(interface_dofs, physical_dofs).astype(np.int32)

        interior_dofs = np.setdiff1d(all_dofs, boundary_dofs).astype(np.int32)

        return {
            'interior': interior_dofs,
            'interface': interface_dofs,
            'physical': physical_dofs
        }

    @staticmethod
    def assemble_system(V, problem_def):
        """
        Helper to extract the system components.
        """
        u, v = TrialFunction(V), TestFunction(V)
        a_form = problem_def.a(u, v)
        L_form = problem_def.L(v)
        A_mat = assemble(a_form)
        b_vec = assemble(L_form)
        return A_mat.array(), b_vec.get_local()

    @staticmethod
    def get_bc_values(g_expr, V, dofs):
        """
        Helper that returns the function values of a expression at given dofs as a vector.
        """
        bc_func = Function(V)
        bc_func.interpolate(g_expr)
        return bc_func.vector().get_local()[dofs]

    @staticmethod
    def reconstruct_solution(V, dofs, u_I_vec, u_G_vec, g_P_vec):
        """
        Helper to build a full dolfin.Function from its DOF vector pieces.
        This version modifies a NumPy array directly for robustness.
        """
        # Create a new, empty NumPy array with the full size of the function space
        full_vec = np.zeros(V.dim())

        # Place the pieces into the correct locations in the NumPy array
        if len(dofs['physical']) > 0:
            full_vec[dofs['physical']] = g_P_vec
        if len(dofs['interface']) > 0:
            full_vec[dofs['interface']] = u_G_vec
        if len(dofs['interior']) > 0:
            full_vec[dofs['interior']] = u_I_vec

        # Create the dolfin.Function and set its entire vector from our completed NumPy array
        u_sol = Function(V)
        u_sol.vector()[:] = full_vec

        return u_sol

    def solve(self, tol, max_iter=100):
        # Initialisation of the variable u_gamma2
        u_G2_vec = np.zeros(len(self.dofs_2['interface']))

        # Function that will contain the interpolation of u_2 on gamma1
        w_1 = Function(self.V_1)

        for k in range(max_iter):
            # compute right hand side of the first equation
            rhs_2 = self.f_bar_2 - self.A_2_gamma2 @ u_G2_vec

            # solving the part with A_omega2^(-1)
            u_I2_vec = np.linalg.solve(self.A_2_interior, rhs_2)

            # put together the solution in a fenics function and interpolate on the boundary
            u_2 = self.reconstruct_solution(self.V_2, self.dofs_2, u_I2_vec, u_G2_vec, self.g_2_vec)
            u_2.set_allow_extrapolation(True)
            w_1.interpolate(u_2)
            # this finally gives the values of u_gamma1
            u_G1_vec = w_1.vector().get_local()[self.dofs_1['interface']]

            # now same process on the other domain to obtain u_gamma2
            rhs_1 = self.f_bar_1 - self.A_1_gamma1 @ u_G1_vec

            u_I1_vec = np.linalg.solve(self.A_1_interior, rhs_1)

            u_1 = self.reconstruct_solution(self.V_1, self.dofs_1, u_I1_vec, u_G1_vec, self.g_1_vec)
            u_1.set_allow_extrapolation(True)
            w_2 = Function(self.V_2)
            w_2.interpolate(u_1)
            u_G2_vec_new = w_2.vector().get_local()[self.dofs_2['interface']]

            error = np.linalg.norm(u_G2_vec_new - u_G2_vec)

            print(f"Iteration {k + 1}: Interface error = {error:.4e}")

            u_G2_vec = u_G2_vec_new

            if error < tol:
                print(f"\nConvergence reached after {k + 1} iterations.")
                rhs_2 = self.f_bar_2 - self.A_2_gamma2 @ u_G2_vec
                u_I2_vec = np.linalg.solve(self.A_2_interior, rhs_2)
                u_2 = self.reconstruct_solution(self.V_2, self.dofs_2, u_I2_vec, u_G2_vec, self.g_2_vec)
                break
        else:
            print("\nWarning: Maximum number of iterations reached.")

        return u_1, u_2


class SchwarzMethod_operator(CompositionMethod):
    def __init__(self, V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                 V_2, mesh_2, boundary_markers_2, problem_def_2, g_2):
        super().__init__(V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                         V_2, mesh_2, boundary_markers_2, problem_def_2, g_2)

        self.dofs_1 = self.get_dof_indices(V_1, boundary_markers_1, 1, 2)
        self.dofs_2 = self.get_dof_indices(V_2, boundary_markers_2, 1, 2)

        self.lambda_1 = np.zeros(len(self.dofs_1["interface"]))
        self.lambda_2 = np.zeros(len(self.dofs_2["interface"]))

        self.fem_solver = fem_solver.FemSolver()

    @staticmethod
    def get_dof_indices(V, boundary_markers, physical_marker, interface_marker):
        """
        Function that returns the indices of dofs lying in interior, on the interface boundary, and
        the physical/true boundary
        :param V: the function space containing the dofs
        :param boundary_markers: mesh function which contains the boundary markers
        :param physical_marker: ID of the physical boundary
        :param interface_marker: ID of the interface boundary
        :return: dict with the needed indices
        """
        bc_interface = DirichletBC(V, Constant(0.0), boundary_markers, interface_marker)
        interface_dofs = np.array(list(bc_interface.get_boundary_values().keys()), dtype=np.int32)

        bc_physical = DirichletBC(V, Constant(0.0), boundary_markers, physical_marker)
        physical_dofs = np.array(list(bc_physical.get_boundary_values().keys()), dtype=np.int32)
        physical_dofs = np.setdiff1d(physical_dofs, interface_dofs).astype(np.int32)
        all_dofs = np.arange(V.dim())

        boundary_dofs = np.union1d(interface_dofs, physical_dofs).astype(np.int32)

        interior_dofs = np.setdiff1d(all_dofs, boundary_dofs).astype(np.int32)

        return {
            'interior': interior_dofs,
            'interface': interface_dofs,
            'physical': physical_dofs
        }

    @staticmethod
    def interpolate_on_gamma(u: Function, V: FunctionSpace, dofs):
        """
        Helper that interpolates the solution on one mesh onto the artificial boundary of the
        other mesh.
        :param u: The solution function on one mesh.
        :param V: The function space corresponding to the other mesh with the artificial boundary
        :param dofs: The dof indices of the artificial boundary.
        :return: A vector containing the function values on the artificial boundary.
        """
        dof_coords = V.tabulate_dof_coordinates()
        interpolated_sol = np.zeros(len(dofs))
        u.set_allow_extrapolation(True)
        for i in range(len(dofs)):
            interpolated_sol[i] = u(dof_coords[dofs[i]])

        return interpolated_sol

    @staticmethod
    def create_bcs(V, boundary_markers, bcs_definitions):
        bcs = []
        for value_expr, target_id in bcs_definitions:
            bcs.append(DirichletBC(V, value_expr, boundary_markers, target_id))
        return bcs

    @staticmethod
    def vector_to_function(vec, V, dofs):
        """
        Helper that writes the values of a vector corresponding to the dofs into a fenics function.
        :param vec: Vector of values.
        :param V: The function space of the fenics function.
        :param dofs: The dof indices corresponding to the values of the vector and the function space V.
        :return: A fenics function with the assigned values.
        """
        u = Function(V)
        u_arr = u.vector().get_local()
        for i in range(len(dofs)):
            u_arr[dofs[i]] = vec[i]
        u.vector().set_local(u_arr)
        u.vector().apply("insert")
        return u

    def solve(self, tol, max_iter=100):
        trial_1 = TrialFunction(self.V_1)
        test_1 = TestFunction(self.V_1)
        trial_2 = TrialFunction(self.V_2)
        test_2 = TestFunction(self.V_2)
        for k in range(max_iter):
            w_1 = self.vector_to_function(self.lambda_1, self.V_1, self.dofs_1["interface"])
            bcs_1 = self.create_bcs(self.V_1, self.boundary_markers_1,
                                    [(self.g_1, 1), (w_1, 2)])
            sol_1 = self.fem_solver.solve(self.V_1, self.problem_def_1.a(trial_1, test_1),
                                          self.problem_def_1.L(test_1), bcs_1)
            sol_1_on_gamma2 = self.interpolate_on_gamma(sol_1, self.V_2, self.dofs_2["interface"])
            self.lambda_2 = sol_1_on_gamma2

            w_2 = self.vector_to_function(self.lambda_2, self.V_2, self.dofs_2["interface"])
            bcs_2 = self.create_bcs(self.V_2, self.boundary_markers_2,
                                    [(self.g_2, 1), (w_2, 2)])
            sol_2 = self.fem_solver.solve(self.V_2, self.problem_def_2.a(trial_2, test_2),
                                          self.problem_def_2.L(test_2), bcs_2)
            sol_2_on_gamma1 = self.interpolate_on_gamma(sol_2, self.V_1, self.dofs_1["interface"])
            self.lambda_1 = sol_2_on_gamma1
        return self.lambda_1, self.lambda_2, sol_1, sol_2


class SchwarzMethod_operator_matrix(CompositionMethod):
    def __init__(self, V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                 V_2, mesh_2, boundary_markers_2, problem_def_2, g_2):
        super().__init__(V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                         V_2, mesh_2, boundary_markers_2, problem_def_2, g_2)

        self.dofs_1 = self.get_dof_indices(V_1, boundary_markers_1, 1, 2)
        self.dofs_2 = self.get_dof_indices(V_2, boundary_markers_2, 1, 2)

        self.lambda_1 = np.zeros(len(self.dofs_1["interface"]))
        self.lambda_2 = np.zeros(len(self.dofs_2["interface"]))

        self.fem_solver = fem_solver.FemSolver()

    @staticmethod
    def get_dof_indices(V, boundary_markers, physical_marker, interface_marker):
        """
        Function that returns the indices of dofs lying in interior, on the interface boundary, and
        the physical/true boundary
        :param V: the function space containing the dofs
        :param boundary_markers: mesh function which contains the boundary markers
        :param physical_marker: ID of the physical boundary
        :param interface_marker: ID of the interface boundary
        :return: dict with the needed indices
        """
        bc_interface = DirichletBC(V, Constant(0.0), boundary_markers, interface_marker)
        interface_dofs = np.array(list(bc_interface.get_boundary_values().keys()), dtype=np.int32)

        bc_physical = DirichletBC(V, Constant(0.0), boundary_markers, physical_marker)
        physical_dofs = np.array(list(bc_physical.get_boundary_values().keys()), dtype=np.int32)
        physical_dofs = np.setdiff1d(physical_dofs, interface_dofs).astype(np.int32)
        all_dofs = np.arange(V.dim())

        boundary_dofs = np.union1d(interface_dofs, physical_dofs).astype(np.int32)

        interior_dofs = np.setdiff1d(all_dofs, boundary_dofs).astype(np.int32)

        return {
            'interior': interior_dofs,
            'interface': interface_dofs,
            'physical': physical_dofs
        }

    @staticmethod
    def interpolate_on_gamma(u: Function, V: FunctionSpace, dofs):
        """
        Helper that interpolates the solution on one mesh onto the artificial boundary of the
        other mesh.
        :param u: The solution function on one mesh.
        :param V: The function space corresponding to the other mesh with the artificial boundary
        :param dofs: The dof indices of the artificial boundary.
        :return: A vector containing the function values on the artificial boundary.
        """
        dof_coords = V.tabulate_dof_coordinates()
        interpolated_sol = np.zeros(len(dofs))
        u.set_allow_extrapolation(True)
        for i in range(len(dofs)):
            interpolated_sol[i] = u(dof_coords[dofs[i]])

        return interpolated_sol

    @staticmethod
    def create_bcs(V, boundary_markers, bcs_definitions):
        bcs = []
        for value_expr, target_id in bcs_definitions:
            bcs.append(DirichletBC(V, value_expr, boundary_markers, target_id))
        return bcs

    @staticmethod
    def vector_to_function(vec, V, dofs):
        """
        Helper that writes the values of a vector corresponding to the dofs into a fenics function.
        :param vec: Vector of values.
        :param V: The function space of the fenics function.
        :param dofs: The dof indices corresponding to the values of the vector and the function space V.
        :return: A fenics function with the assigned values.
        """
        u = Function(V)
        u_arr = u.vector().get_local()
        for i in range(len(dofs)):
            u_arr[dofs[i]] = vec[i]
        u.vector().set_local(u_arr)
        u.vector().apply("insert")
        return u

    def solve_subdomain_get_interface_values(self, lambda_vec, domain_idx,
                                             use_homogeneous_bc=False, use_homogeneous_f=False):
        """
        Solves the problem for given lambda value on artificial boundary and in domain with
        given domain index. Corresponds to the operator P in the notes.
        :param lambda_vec: Value on artificial boundary
        :param domain_idx: Index of the domain
        :return:
        """
        if domain_idx == 1:
            V, boundary_markers, problem_def = self.V_1, self.boundary_markers_1, self.problem_def_1
            # Use original 'g' or a zero constant based on the flag
            g = self.g_1 if not use_homogeneous_bc else Constant(0.0)
            f_expr = problem_def.f if not use_homogeneous_f else Constant(0.0)
            dofs_source, dofs_target = self.dofs_1, self.dofs_2
            V_target = self.V_2
        else:
            V, boundary_markers, problem_def = self.V_2, self.boundary_markers_2, self.problem_def_2
            # Use original 'g' or a zero constant based on the flag
            g = self.g_2 if not use_homogeneous_bc else Constant(0.0)
            f_expr = problem_def.f if not use_homogeneous_f else Constant(0.0)
            dofs_source, dofs_target = self.dofs_2, self.dofs_1
            V_target = self.V_1

        w = self.vector_to_function(lambda_vec, V, dofs_source["interface"])
        bcs = self.create_bcs(V, boundary_markers, [(g, 1), (w, 2)])
        trial, test = TrialFunction(V), TestFunction(V)
        a_form = problem_def.a(trial, test)
        L_form = f_expr * test * dx
        u_sol = self.fem_solver.solve(V, a_form, L_form, bcs)
        u_sol_on_gamma = self.interpolate_on_gamma(u_sol, V_target, dofs_target["interface"])
        return u_sol_on_gamma


    def apply_operator(self, lambda_full_vec):
        """
        Applies the operator P to implement the matrix operator and to compute A lambda f.
        :param lambda_full_vec:
        :return:
        """
        n_1 = len(self.dofs_1["interface"])
        self.lambda_1 = lambda_full_vec[:n_1]
        self.lambda_2 = lambda_full_vec[n_1:]

        P_1 = self.solve_subdomain_get_interface_values(self.lambda_1, 1,
                                      True, True)
        result_2 = self.lambda_2 - P_1

        P_2 = self.solve_subdomain_get_interface_values(self.lambda_2, 2,
                                      True, True)
        result_1 = self.lambda_1 - P_2

        return np.concatenate([result_1, result_2])

    def construct_solution(self, lambda_solution, domain_idx):
        if domain_idx == 1:
            V, boundary_markers, problem_def, g = self.V_1, self.boundary_markers_1, self.problem_def_1, self.g_1
            dofs_source = self.dofs_1
        else:
            V, boundary_markers, problem_def, g = self.V_2, self.boundary_markers_2, self.problem_def_2, self.g_2
            dofs_source = self.dofs_2

        w = self.vector_to_function(lambda_solution, V, dofs_source["interface"])
        bcs = self.create_bcs(V, boundary_markers, [(g, 1), (w, 2)])
        trial, test = TrialFunction(V), TestFunction(V)

        u_final = self.fem_solver.solve(V, problem_def.a(trial, test), problem_def.L(test), bcs)
        return u_final

    def solve(self, tol, max_iter=100):

        f_1 = self.solve_subdomain_get_interface_values(np.zeros_like(self.lambda_2), 2)
        f_2 = self.solve_subdomain_get_interface_values(np.zeros_like(self.lambda_1), 1)

        f = np.concatenate([f_1, f_2])

        size = len(self.dofs_1["interface"]) + len(self.dofs_2["interface"])

        A_operator = LinearOperator((size, size), self.apply_operator)

        residuals = []
        def callback(res):
            residuals.append(res)
            print(f"GMRES Iteration {len(residuals)}, Relative Residual = {res:.4e}")

        lambda_solution, exit_code = gmres(A_operator, f,
                                           maxiter=max_iter, callback=callback, restart=100)#, callback_type='pr_norm')
        n_1 = len(self.dofs_1["interface"])
        lambda_1_sol = lambda_solution[:n_1]
        lambda_2_sol = lambda_solution[n_1:]

        u_1_final = self.construct_solution(lambda_1_sol, 1)
        u_2_final = self.construct_solution(lambda_2_sol, 2)
        return u_1_final, u_2_final