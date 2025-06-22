from ufl import TrialFunction

import ProblemDefinition as problem_def
from dolfin import *
import FemSolver as fem_solver
import numpy as np
from HelperModules import visualiser, analyser, geometry_parser

# Set up of the mesh using helpers
visualiser = visualiser.Visualiser()
solver = fem_solver.FemSolver()
geo_parser = geometry_parser.GeometryParser(0.1)
geo_parser.circle_mesh((0, 0), 1, "circle")
geo_parser.rectangle_mesh((0, 0), 0.5, 2, "rectangle")

#mesh = geo_parser.load_mesh("circle")
#mesh = geo_parser.load_mesh("rectangle")
mesh = UnitSquareMesh(20, 20)

"""
# Helmholtz problem 
V = FunctionSpace(mesh, "CG", 1)
u0 = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=6)
f = Expression("(8*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2)
bcs_definition = [(u0, 0)]
boundary_markers = MeshFunction("size_t", mesh, 1)
boundary_markers.set_all(0)
bcs = [DirichletBC(V, u0, boundary_markers, 0)]
helmholtz = problem_def.HelmholtzProblem(f)
u = TrialFunction(V)
v = TestFunction(V)
a_form = helmholtz.a(u, v)
L_form = helmholtz.L(v)
u_sol = solver.solve(V, a_form, L_form, bcs)
visualiser.mesh_plot([mesh], True)
visualiser.plot_3d(u_sol, mesh)
"""

"""
# Poisson equation
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0) and on_boundary

left = Left()
right = Right()
boundary_markers = MeshFunction("size_t", mesh, 1)
boundary_markers.set_all(0)
left.mark(boundary_markers, 1)
right.mark(boundary_markers, 2)
V = FunctionSpace(mesh, "CG", 1)
#u0 = Expression("1 + x[0]*x[0] + 2 * x[1]*x[1]", degree=2)
u = TrialFunction(V)
v = TestFunction(V)
u1 = Constant(0)
u2 = Constant(-2)
f = Constant(-6)
bcs_definition = [(u1, 1), (u2, 2)]
bcs = []
for value_expr, target_id in bcs_definition:
    bcs.append(DirichletBC(V, value_expr, boundary_markers, target_id))

poisson = problem_def.PoissonProblem(f)
a_form = poisson.a(u, v)
L_form = poisson.L(v)
u_sol = solver.solve(V, a_form, L_form, bcs)
"""

#visualiser.mesh_plot([mesh], True)
#visualiser.plot_3d(u_sol, mesh)


"""
# Linear elasticity problem, currently only works for built in meshes
V = VectorFunctionSpace(mesh, "CG", 1, 2)
g_x = "pow(x[0], 6)"
g_y = "pow(x[1], 6)"
g = Expression((g_x, g_y), degree=2)
lin_elas = problem_def.LinearElasticity(g)

u_sol = solver.solve(lin_elas, V)
#visualiser.mesh_plot([mesh], True)
#visualiser.plot_3d(u_sol, mesh)
"""

"""
# convergence result
V = FunctionSpace(mesh, "CG", 1)
u0 = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=6)
f = Expression("(8*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2)
helmholtz = problem_def.HelmholtzProblem(f)
polynomial_degrees = [1, 2]
step_sizes = np.logspace(np.log10(np.sqrt(0.001)), np.log10(np.sqrt(0.1)), num=4)
analyser = analyser.Analyser(helmholtz, u0)
results = analyser.run_analysis(polynomial_degrees, step_sizes, u0)
visualiser.convergence_rates_plot(results)
"""
