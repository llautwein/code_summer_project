import ProblemDefinition
import FemSolver
from dolfin import *

class Analyser:
    """
    Verifies the convergence results for different stepsizes and polynomial degrees.
    """
    def __init__(self, problem: ProblemDefinition):
        self.problem = problem
        self.results = {}

    def run_analysis(self, polynomial_degrees, step_sizes, u_exact):
        solver = FemSolver.FemSolver()
        for d in polynomial_degrees:
            errors_L2 = []
            errors_H1 = []

            for h in step_sizes:
                n = int(round(1/h))
                print(f"Degree d={d}, step size h={h}, n={n}")
                mesh = UnitSquareMesh(n, n)

                u_h = solver.solve(self.problem, mesh, d)

                errors_L2.append(errornorm(u_exact, u_h, 'L2', mesh=mesh))
                errors_H1.append(errornorm(u_exact, u_h, 'H1', mesh=mesh))

            self.results[d] = {'h': step_sizes, 'L2': errors_L2, 'H1': errors_H1}
        return self.results
