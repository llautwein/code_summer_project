import ProblemDefinition as problem_definition
import FemSolver as fem_solver
from dolfin import *

class CompositionMethod:
    def __init__(self, V_1, mesh_1, boundary_markers_1, problem_def_1, g_1,
                 V_2, mesh_2, boundary_markers_2, problem_def_2, g_2):
        self.mesh_1 = mesh_1
        self.boundary_markers_1 = boundary_markers_1
        self.problem_def_1 = problem_def_1
        self.V_1 = V_1
        self.g_1 = g_1   # original boundary condition of problem 1
        self.mesh_2 = mesh_2
        self.boundary_markers_2 = boundary_markers_2
        self.problem_def_2 = problem_def_2
        self.V_2 = V_2
        self.g_2 = g_2   # original boundary condition of problem 2
        self.fem_solver = fem_solver.FemSolver()

    def solve(self, tol, max_iter):
        pass

class SchwarzMethod_primitive(CompositionMethod):
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
                print(f"Tolerance reached after {k} iteration.")
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

