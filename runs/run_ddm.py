from HelperModules import GeometryParser, visualiser
from dolfin import *
import ProblemDefinition as problem_def
import CompositionMethod as cm
import InterfaceHandler as ih

p0 = Point(0.25, 0.25)
length = 0.5
height = 1
mid_intersection = 0.75
delta = 0.1
gmsh_parameters = {"refine_at_interface": True,
                   "refinement_factor": 10,
                   "transition_ratio": 0.1}

geo_parser = GeometryParser.GeometryParser()
mesh_upper, mesh_lower = geo_parser.create_offset_meshes(p0, length, height, mid_intersection, delta, 0.5, 0.1, gmsh_parameters=gmsh_parameters)

interface_handler = ih.InterfaceHandler(mesh_upper, mesh_lower)
y_interface_of_upper_domain = mid_intersection - delta / 2
y_interface_of_lower_domain = mid_intersection + delta / 2
boundary_markers_upper, boundary_markers_lower = interface_handler.mark_interface_boundaries(2)

File("output_files/boundary_markers.pvd") << boundary_markers_upper

V_1 = FunctionSpace(mesh_upper, "CG", 1)
V_2 = FunctionSpace(mesh_lower, "CG", 1)
g_1 = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=6)
f_1 = Expression("(8*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2)
sol_analytic_rec_upper = interpolate(g_1, V_1)
sol_analytic_rec_lower = interpolate(g_1, V_2)

model_problem = problem_def.ModelProblem(f_1)

vs = visualiser.Visualiser()
vs.mesh_plot([mesh_upper, mesh_lower], True)

# Select version of the Schwarz method (alternating, matrix free, algebraic)
schwarz_algorithm = cm.SchwarzMethodAlgebraic(V_1, mesh_upper, boundary_markers_upper, model_problem, g_1,
                                              V_2, mesh_lower, boundary_markers_lower, model_problem, g_1, True)
u1, u2 = schwarz_algorithm.solve(1e-4, 100)
print(f"Error of u1: {errornorm(sol_analytic_rec_upper, u1,'L2', mesh=mesh_upper)}")
print(f"Error of u2: {errornorm(sol_analytic_rec_lower, u2,'L2', mesh=mesh_lower)}")
vs.heatmap_plot(u1, mesh_upper)
vs.heatmap_plot(u2, mesh_lower)

