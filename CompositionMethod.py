import FemSolver as fem_solver
from dolfin import *
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator, spsolve, splu
from scipy.sparse import csc_matrix


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

    @staticmethod
    def get_dof_indices(V, boundary_markers, physical_marker, interface_marker):
        """
        Function that returns the indices of dofs lying in interior, on the interface boundary, and
        the physical/true boundary
        :param V: the function space containing the dofs
        :param boundary_markers: mesh function which contains the boundary markers
        :param physical_marker: ID of the physical boundary (1)
        :param interface_marker: ID of the interface boundary (2)
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
    def create_bcs(V, boundary_markers, bcs_definitions):
        """
        Helper that creates a list of Dirichlet boundary conditions. Used two define BCs on the
        true and artificial boundary.
        :param V: The function space corresponding to the boundary conditions.
        :param boundary_markers: A mesh function which marks the boundary.
        :param bcs_definitions: List of pairs in the form (expr, boundary_idx) [1: true, 2: artificial]
        :return: List of Dirichlet boundary conditions.
        """
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

    def solve(self, tol, max_iter):
        pass


class SchwarzMethodMatrixFree(CompositionMethod):
    """
    Solves the interface problem without explicitly assembling the stiffness matrix.
    The PDEs are solved using fenics internal solve, the interface problem is solved with GMRES.
    """
    def __init__(self, V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                 V_2, mesh_2, boundary_markers_2, problem_def_2, g_2):
        super().__init__(V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                         V_2, mesh_2, boundary_markers_2, problem_def_2, g_2)

        # Identify the degrees of freedom (DoFs) for the interior, physical boundary,
        # and artificial interface for each subdomain.
        self.dofs_1 = self.get_dof_indices(V_1, boundary_markers_1, 1, 2)
        self.dofs_2 = self.get_dof_indices(V_2, boundary_markers_2, 1, 2)

        self.fem_solver = fem_solver.FemSolver()

    @staticmethod
    def interpolate_on_gamma(u: Function, V: FunctionSpace, dofs):
        """
        Helper that evaluates the solution 'u' from one mesh at the DoF coordinates
        of the interface on the other mesh.
        :param u: The solution function on one mesh.
        :param V: The function space corresponding to the other mesh with the artificial boundary
        :param dofs: The dof indices of the artificial boundary corresponding to V.
        :return: A vector containing the function values on the artificial boundary.
        """
        dof_coords = V.tabulate_dof_coordinates()
        interpolated_sol = np.zeros(len(dofs))
        u.set_allow_extrapolation(True)
        for i in range(len(dofs)):
            interpolated_sol[i] = u(dof_coords[dofs[i]])

        return interpolated_sol


    def solve_subdomain_get_interface_values(self, lambda_vec, domain_idx,
                                             use_homogeneous_bc=False, use_homogeneous_f=False):
        """
        This function solves the PDE on one subdomain and returns the interpolation of this solution
        on the other subdomain's interface.
        :param lambda_vec: The given values on the artificial boundary of the subdomain.
        :param domain_idx: The index of the subdomain (1 or 2) to solve on.
        :param use_homogeneous_bc: Boolean, if true sets physical boundary condition g=0.
        :param use_homogeneous_f: Boolean, if true sets the rhs term of the PDE f=0.
        :return: Interpolation of the solution on the other subdomain's interface.
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

        # Create a Function on the source interface from the lambda_vec data
        w = self.vector_to_function(lambda_vec, V, dofs_source["interface"])
        # Combine boundary data
        bcs = self.create_bcs(V, boundary_markers, [(g, 1), (w, 2)])
        # Set up the problem to solve
        trial, test = TrialFunction(V), TestFunction(V)
        a_form = problem_def.a(trial, test)
        L_form = f_expr * test * dx
        # Solve the PDE on the subdomain
        u_sol = self.fem_solver.solve(V, a_form, L_form, bcs)
        # Interpolate the solution on the other subdomain's interface boundary
        u_sol_on_gamma = self.interpolate_on_gamma(u_sol, V_target, dofs_target["interface"])
        return u_sol_on_gamma


    def apply_operator(self, lambda_full_vec):
        """
        Applies the operator P to implement the matrix operator and to compute A lambda = b.
        :param lambda_full_vec: The vector of unknowns lambda = [lambda_1, lambda_2].
        :return: The product of the operator matrix with lambda: A*[lambda_1, lambda_2].
        """
        n_1 = len(self.dofs_1["interface"])
        lambda_1 = lambda_full_vec[:n_1]
        lambda_2 = lambda_full_vec[n_1:]

        P_2 = self.solve_subdomain_get_interface_values(lambda_2, 2,
                                                        True, True)
        result_1 = lambda_1 - P_2

        P_1 = self.solve_subdomain_get_interface_values(lambda_1, 1,
                                      True, True)
        result_2 = lambda_2 - P_1

        return np.concatenate([result_1, result_2])

    def construct_solution(self, lambda_solution, domain_idx):
        """
        After finding the correct interface values (lambda_solution), this function
        computes the final solution in the specified subdomain.
        :param lambda_solution: The final interface values on one subdomain.
        :param domain_idx: The index of the subdomain (1 or 2).
        :return: The final solution as a fenics function.
        """
        if domain_idx == 1:
            V, boundary_markers, problem_def, g = self.V_1, self.boundary_markers_1, self.problem_def_1, self.g_1
            dofs_source = self.dofs_1
        else:
            V, boundary_markers, problem_def, g = self.V_2, self.boundary_markers_2, self.problem_def_2, self.g_2
            dofs_source = self.dofs_2

        # Use the final, correct lambda values on the interface
        w = self.vector_to_function(lambda_solution, V, dofs_source["interface"])
        # Use the original physical boundary conditions
        bcs = self.create_bcs(V, boundary_markers, [(g, 1), (w, 2)])
        # Set up the problem and solve
        trial, test = TrialFunction(V), TestFunction(V)
        u_final = self.fem_solver.solve(V, problem_def.a(trial, test), problem_def.L(test), bcs)
        return u_final

    def solve(self, tol, max_iter=100, restart=20):

        # Compute the rhs of the final system
        lambda_1_init = np.zeros(len(self.dofs_1["interface"]))
        lambda_2_init = np.zeros(len(self.dofs_2["interface"]))
        f_1 = self.solve_subdomain_get_interface_values(np.zeros_like(lambda_2_init), 2)
        f_2 = self.solve_subdomain_get_interface_values(np.zeros_like(lambda_1_init), 1)

        f = np.concatenate([f_1, f_2])

        size = len(self.dofs_1["interface"]) + len(self.dofs_2["interface"])

        A_operator = LinearOperator((size, size), self.apply_operator)

        residuals = []
        def callback(res):
            residuals.append(res)
            print(f"GMRES Iteration {len(residuals)}, Relative Residual = {res:.4e}")

        lambda_solution, exit_code = gmres(A_operator, f, maxiter=max_iter,
                                           callback=callback, restart=restart)
        n_1 = len(self.dofs_1["interface"])
        lambda_1_sol = lambda_solution[:n_1]
        lambda_2_sol = lambda_solution[n_1:]

        u_1_final = self.construct_solution(lambda_1_sol, 1)
        u_2_final = self.construct_solution(lambda_2_sol, 2)
        return u_1_final, u_2_final


class SchwarzMethodAlgebraic(CompositionMethod):
    """
    Solves the interface problem using an algebraic approach.
    Subdomain matrices are explicitly assembled as sparse matrices, and
    interior solvers are pre-factorized for efficiency.
    """
    def __init__(self, V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                 V_2, mesh_2, boundary_markers_2, problem_def_2, g_2):
        super().__init__(V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                         V_2, mesh_2, boundary_markers_2, problem_def_2, g_2)

        # pre computations for domain 1
        self.dofs_1 = self.get_dof_indices(V_1, boundary_markers_1, 1, 2)
        self.A_1, self.f_1 = self.assemble_system(V_1, problem_def_1)

        self.A_1_interior = self.A_1[self.dofs_1["interior"], :][:, self.dofs_1["interior"]].tocsc()
        self.A_1_gamma1 = self.A_1[self.dofs_1["interior"], :][:, self.dofs_1["interface"]].tocsc()
        self.A_1_phys1 = self.A_1[self.dofs_1["interior"], :][:, self.dofs_1["physical"]].tocsc()

        self.g_1_vec = self.get_bc_values(g_1, V_1, self.dofs_1["physical"])

        self.solver_1 = splu(self.A_1_interior)

        # pre computations for domain 2
        self.dofs_2 = self.get_dof_indices(V_2, boundary_markers_2, 1, 2)
        self.A_2, self.f_2 = self.assemble_system(V_2, problem_def_2)

        self.A_2_interior = self.A_2[self.dofs_2["interior"], :][:, self.dofs_2["interior"]].tocsc()
        self.A_2_gamma2 = self.A_2[self.dofs_2["interior"], :][:, self.dofs_2["interface"]].tocsc()
        self.A_2_phys2 = self.A_2[self.dofs_2["interior"], :][:, self.dofs_2["physical"]].tocsc()

        self.g_2_vec = self.get_bc_values(g_2, V_2, self.dofs_2["physical"])

        self.solver_2 = splu(self.A_2_interior)

    @staticmethod
    def assemble_system(V, problem_def):
        """
        Helper to extract the system components. Uses scipy.sparse to store the matrices.
        """
        u, v = TrialFunction(V), TestFunction(V)
        a_form = problem_def.a(u, v)
        L_form = problem_def.L(v)
        A_mat = assemble(a_form)
        b_vec = assemble(L_form)
        A_petsc = as_backend_type(A_mat).mat()

        A_scipy_sparse = csc_matrix(A_petsc.getValuesCSR()[::-1], shape=A_petsc.size)

        return A_scipy_sparse, b_vec.get_local()

    @staticmethod
    def get_bc_values(g_expr, V, dofs):
        """
        Helper that returns the function values of an expression at given dofs as a vector.
        """
        bc_func = Function(V)
        bc_func.interpolate(g_expr)
        return bc_func.vector().get_local()[dofs]

    @staticmethod
    def build_dolfin_function_from_dofs(V, dofs, interior_vals, interface_vals, physical_vals):
        """
        Helper to build a full dolfin.Function from its DOF vector pieces.
        """
        # Create a new, empty NumPy array with the full size of the function space
        full_vec = np.zeros(V.dim())
        # Place the pieces into the correct locations in the NumPy array
        if len(dofs['physical']) > 0:
            full_vec[dofs['physical']] = physical_vals
        if len(dofs['interface']) > 0:
            full_vec[dofs['interface']] = interface_vals
        if len(dofs['interior']) > 0:
            full_vec[dofs['interior']] = interior_vals

        # Create the dolfin.Function and set its entire vector from our completed NumPy array
        u_sol = Function(V)
        u_sol.vector()[:] = full_vec

        return u_sol

    @staticmethod
    def interpolate_on_gamma(u: Function, V: FunctionSpace, dofs):
        """
        Helper that evaluates the solution 'u' from one mesh at the dof coordinates
        of the interface on the other mesh.
        """
        dof_coords = V.tabulate_dof_coordinates()
        interpolated_sol = np.zeros(len(dofs))
        u.set_allow_extrapolation(True)
        for i, dof_index in enumerate(dofs):
            point = dof_coords[dof_index]
            interpolated_sol[i] = u(point)
        return interpolated_sol

    def solve_subdomain_and_interpolate(self, domain_idx, lambda_vec,
                                        use_homogeneous_f=False, use_homogeneous_g=False):
        """
        Solves the PDE on a given subdomain and returns the values of the solution
        on the other subdomain's interface.
        Can be configured to use homogeneous source terms (f=0) and physical
        boundary conditions (g=0), which is needed for applying the operator.
        :param domain_idx: The index of the subdomain (1 or 2) to solve on.
        :param lambda_vec: The Dirichlet data for the interface of the source domain.
        :param use_homogeneous_f: If True, use f=0 for the PDE source term.
        :param use_homogeneous_g: If True, use g=0 for the physical boundary.
        :return: A numpy vector of the solution values on the target domain's interface.
        """
        if domain_idx == 1:
            # Select matrices and vectors for domain 1
            A_interior, A_gamma = self.A_1_interior, self.A_1_gamma1
            solver = self.solver_1
            A_phys = self.A_1_phys1
            f_interior_full = self.f_1[self.dofs_1["interior"]]
            g_vec_full = self.g_1_vec
            V_source, dofs_source = self.V_1, self.dofs_1
            V_target, dofs_target = self.V_2, self.dofs_2
        elif domain_idx == 2:
            # Select matrices and vectors for domain 2
            A_interior, A_gamma = self.A_2_interior, self.A_2_gamma2
            solver = self.solver_2
            A_phys = self.A_2_phys2
            f_interior_full = self.f_2[self.dofs_2["interior"]]
            g_vec_full = self.g_2_vec
            V_source, dofs_source = self.V_2, self.dofs_2
            V_target, dofs_target = self.V_1, self.dofs_1
        else:
            raise ValueError("domain_idx must be 1 or 2")

        # Determine the correct f and g vectors based on flags
        f_interior = f_interior_full if not use_homogeneous_f else np.zeros_like(f_interior_full)
        g_vec = g_vec_full if not use_homogeneous_g else np.zeros_like(g_vec_full)

        # Calculate the effective right-hand side for the interior problem
        f_bar = f_interior - A_phys @ g_vec

        # Solve for the interior degrees of freedom
        rhs = f_bar - A_gamma @ lambda_vec
        u_interior = solver.solve(rhs)

        # Reconstruct the full solution
        u_full_func = self.build_dolfin_function_from_dofs(V_source, dofs_source, u_interior,
                                                lambda_vec, g_vec)

        # Interpolate the solution onto the target domain's function space
        u_interpolated_on_gamma = self.interpolate_on_gamma(u_full_func, V_target, dofs_target["interface"])
        # Extract and return the values on the target interface
        return u_interpolated_on_gamma


    def apply_operator(self, lambda_full_vec):

        n_1 = len(self.dofs_1["interface"])
        lambda_1 = lambda_full_vec[:n_1]
        lambda_2 = lambda_full_vec[n_1:]

        P_2_lambda = self.solve_subdomain_and_interpolate(2, lambda_2, True, True)
        result_1 = lambda_1 - P_2_lambda

        P_1_lambda = self.solve_subdomain_and_interpolate(1, lambda_1, True, True)
        results_2 = lambda_2 - P_1_lambda

        return np.concatenate([result_1, results_2])

    def construct_final_solution(self, lambda_vec, domain_idx):
        if domain_idx == 1:
            # Select matrices and vectors for domain 1
            A_interior, A_gamma = self.A_1_interior, self.A_1_gamma1
            solver = self.solver_1
            A_phys = self.A_1_phys1
            f_interior_full = self.f_1[self.dofs_1["interior"]]
            g_vec = self.g_1_vec
            V, dofs = self.V_1, self.dofs_1
        elif domain_idx == 2:
            # Select matrices and vectors for domain 2
            A_interior, A_gamma = self.A_2_interior, self.A_2_gamma2
            solver = self.solver_2
            A_phys = self.A_2_phys2
            f_interior_full = self.f_2[self.dofs_2["interior"]]
            g_vec = self.g_2_vec
            V, dofs = self.V_2, self.dofs_2
        else:
            raise ValueError("domain_idx must be 1 or 2")

        rhs = f_interior_full - A_phys @ g_vec - A_gamma @ lambda_vec
        u_interior = solver.solve(rhs)

        # Reconstruct the full solution
        u_func = self.build_dolfin_function_from_dofs(V, dofs, u_interior,
                                                      lambda_vec, g_vec)
        return u_func

    def solve(self, tol, max_iter=100, restart=20):
        # Set up of the right hand side corresponding to the first row
        rhs1 = self.solve_subdomain_and_interpolate(2, np.zeros_like(self.dofs_2["interface"]))
        # Set up of the right hand side corresponding to the second row
        rhs2 = self.solve_subdomain_and_interpolate(1, np.zeros_like(self.dofs_1["interface"]))

        rhs = np.concatenate([rhs1, rhs2])

        size = len(self.dofs_1["interface"]) + len(self.dofs_2["interface"])

        A_operator = LinearOperator((size, size), self.apply_operator)

        residuals = []
        def callback(res):
            residuals.append(res)
            print(f"GMRES iteration {len(residuals)}, relative residual = {res:.4e}")

        lambda_solution, exit_code = gmres(A_operator, rhs, maxiter=max_iter,
                                           callback=callback, restart=20)
        n_1 = len(self.dofs_1["interface"])
        lambda_1_sol = lambda_solution[:n_1]
        u1 = self.construct_final_solution(lambda_1_sol, 1)
        lambda_2_sol = lambda_solution[n_1:]
        u2 = self.construct_final_solution(lambda_2_sol, 2)

        return u1, u2



class SchwarzMethodAlternating(CompositionMethod):
    """
    Implements the classical alternating Schwarz method using fenics to solve the PDEs on each
    subdomain.
    """
    def __init__(self, V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                 V_2, mesh_2, boundary_markers_2, problem_def_2, g_2):
        super().__init__(V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                         V_2, mesh_2, boundary_markers_2, problem_def_2, g_2)


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


