from HelperModules import geometry_parser, visualiser
from dolfin import *
import ProblemDefinition as problem_def
import CompositionMethod as cm
import InterfaceHandler as ih

geo_parser = geometry_parser.GeometryParser(0.1)
geo_parser.rectangle_mesh((0, 0), 1, 1, "rectangle")
geo_parser.circle_mesh((1, 1), 0.5, "circle")
mesh_rectangle = geo_parser.load_mesh("rectangle")
#mesh_rectangle = UnitSquareMesh(5, 5)
mesh_circle = geo_parser.load_mesh("circle")

interface_handler = ih.InterfaceHandler(mesh_rectangle, mesh_circle)

boundary_markers_1, boundary_markers_2 = interface_handler.mark_interface_boundaries()

File("boundary_markers.pvd") << boundary_markers_1

V_1 = FunctionSpace(mesh_rectangle, "CG", 1)
g_1 = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=6)
f_1 = Expression("(8*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2)
sol_analytic_rec = interpolate(g_1, V_1)

V_2 = FunctionSpace(mesh_circle, "CG", 1)
sol_analytic_circ = interpolate(g_1, V_2)
g_2 = Expression("1 + x[0]*x[0] + 2 * x[1]*x[1]", degree=2)
f_2 = Constant(-6.0)

model_problem = problem_def.ModelProblem(f_1)
poisson = problem_def.PoissonProblem(f_2)

vs = visualiser.Visualiser()
vs.mesh_plot([mesh_rectangle, mesh_circle], True)


# Select version of the Schwarz method (alternating, matrix free, algebraic)
schwarz_algorithm = cm.SchwarzMethodMatrixFree(V_1, mesh_rectangle, boundary_markers_1, model_problem, g_1,
                 V_2, mesh_circle, boundary_markers_2, model_problem, g_1)
u1, u2 = schwarz_algorithm.solve(1e-4, 100)

vs.heatmap_plot(sol_analytic_rec, mesh_rectangle)
vs.heatmap_plot(sol_analytic_circ, mesh_circle)
vs.heatmap_plot(u1, mesh_rectangle)
vs.heatmap_plot(u2, mesh_circle)

