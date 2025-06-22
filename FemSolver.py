import ProblemDefinition
from dolfin import *

class FemSolver:
    """
    Class that solves a linear variational problem for a given problem formulation
    """
    def solve(self, V, a_form, L_form, bcs):
        u_sol = Function(V)
        solve(a_form == L_form, u_sol, bcs)
        return u_sol

