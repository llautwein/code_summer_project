import ProblemDefinition
from dolfin import *

class FemSolver:
    """
    Class that solves a linear variational problem for a given problem formulation
    """

    @staticmethod
    def g_boundary(x, on_boundary):
        return on_boundary

    def solve(self, problem: ProblemDefinition, V):
        g = problem.g
        bc = DirichletBC(V, g, self.g_boundary)

        u = TrialFunction(V)
        v = TestFunction(V)
        a = problem.a(u, v)
        L = problem.L(v)

        u_sol = Function(V)
        solve(a == L, u_sol, bc)
        return u_sol

