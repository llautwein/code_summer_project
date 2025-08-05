from HelperModules import GeometryParser as gp
import InterfaceHandler as ih
from HelperModules import visualiser as vis
import numpy as np
import ProblemDefinition as problem_def
import CompositionMethod as cm
from dolfin import *
import fenics

p0 = Point(0, -0.25)
length = 1
height = 2
width = 1
mid_intersection = 0.75
delta = 0.1
delta_2_pctg = 0.9
h = 0.2
mesh_option = "gmsh"
gmsh_parameters = {"refine_at_interface": True,
                    "refinement_factor": 10.0,
                    "transition_ratio": 0.1}
visualiser = vis.Visualiser()
geo_parser = gp.GeometryParser()
cuboid_upper, cuboid_lower = geo_parser.create_3d_meshes(p0, length, width, height, mid_intersection, delta, h)

interface_handler = ih.OverlappingRectanglesInterfaceHandler(cuboid_upper, cuboid_lower)
y_interface_of_upper_domain = mid_intersection - delta / 2
y_interface_of_lower_domain = mid_intersection + delta / 2
bm_1, bm_2 = interface_handler.mark_interface_boundaries_3d(y_interface_of_upper_domain, y_interface_of_lower_domain)
File("output_files/boundary_markers.pvd") << bm_1

V_1 = FunctionSpace(cuboid_upper, "CG", 1)
V_2 = FunctionSpace(cuboid_lower, "CG", 1)
g_1 = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])*sin(2*pi*x[2])", degree=6)
f_1 = Expression("(12*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])*sin(2*pi*x[2])", degree=2)
sol_analytic_rec_upper = interpolate(g_1, V_1)
sol_analytic_rec_lower = interpolate(g_1, V_2)

model_problem = problem_def.ModelProblem(f_1)
schwarz_algorithm = cm.SchwarzMethodMatrixFree(V_1, cuboid_upper, bm_1, model_problem, g_1,
                                                V_2, cuboid_lower, bm_2, model_problem, g_1)
u1, u2 = schwarz_algorithm.solve(1e-4, 100)
print(f"Error of u1: {errornorm(sol_analytic_rec_upper, u1,'L2', mesh=cuboid_upper)}")
print(f"Error of u2: {errornorm(sol_analytic_rec_lower, u2,'L2', mesh=cuboid_lower)}")
"""
visualiser.mesh_plot([rec_upper, rec_lower])
visualiser.mesh_plot([rec_upper])
visualiser.mesh_plot([rec_lower])
#interface_handler = ih.InterfaceHandler(mesh_circle, mesh_rectangle)
#bm_1, bm_2 = interface_handler.mark_interface_boundaries()
#File("output_files/boundary_markers.pvd") << bm_1
"""
"""
p0 = Point(0, -0.25)
length = 1
height = 2
mid_intersection = 0.75
delta = 0.2
h = 0.1
mesh_rectangle_1, mesh_rectangle_2 = geo_parser.create_conforming_meshes(p0, length, height, mid_intersection, delta)

interface_handler = ih.OverlappingRectanglesInterfaceHandler(mesh_rectangle_1, mesh_rectangle_2)
y_interface_of_upper_domain = mid_intersection - delta / 2
y_interface_of_lower_domain = mid_intersection + delta / 2
boundary_markers_1, boundary_markers_2 = interface_handler.mark_interface_boundaries(
    y_interface_of_upper_domain,
    y_interface_of_lower_domain
)

File("output_files/boundary_markers.pvd") << boundary_markers_2


visualiser.mesh_plot([mesh_rectangle_1, mesh_rectangle_2])
visualiser.mesh_plot([mesh_rectangle_1])
visualiser.mesh_plot([mesh_rectangle_2])
"""