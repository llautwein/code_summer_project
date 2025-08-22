from HelperModules import GeometryParser as gp
import InterfaceHandler as ih
from HelperModules import visualiser as vis
from HelperModules import analyser
import numpy as np
import ProblemDefinition as problem_def
import CompositionMethod as cm
from dolfin import *
import fenics

p0 = Point(0, 0, -0.25)
length = 1
height = 2
width = 1
mid_intersection = 0.75
delta = 0.1
delta_2_pctg = 0.9
h = 0.3
mesh_option = "gmsh"
gmsh_parameters = {"refine_at_interface": True,
                    "refinement_factor": 10.0,
                    "transition_ratio": 0.1}
visualiser = vis.Visualiser()
geo_parser = gp.GeometryParser()
analyser = analyser.Analyser()
rec_upper, rec_lower = geo_parser.create_3d_meshes(p0, length, width, height, mid_intersection, delta, h,
                                                    gmsh_parameters=gmsh_parameters)
exit()

#visualiser.mesh_plot([rec_upper, rec_lower], ax_equal=True)
#visualiser.mesh_plot([rec_upper], ax_equal=True)
#visualiser.mesh_plot([rec_lower], ax_equal=True)


interface_handler = ih.OverlappingRectanglesInterfaceHandler(rec_upper, rec_lower)
y_interface_of_upper_domain = mid_intersection - delta / 2  # Bottom face of upper cuboid
y_interface_of_lower_domain = mid_intersection + delta / 2  # Top face of lower cuboid

bm_1, bm_2 = interface_handler.mark_interface_boundaries(
    y_interface_of_upper_domain,
    y_interface_of_lower_domain
)
File("output_files/boundary_markers.pvd") << bm_1
V_1 = FunctionSpace(rec_upper, "CG", 1)
V_2 = FunctionSpace(rec_lower, "CG", 1)
g_1 = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=6)
f_1 = Expression("(8*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2)
sol_analytic_rec_upper = interpolate(g_1, V_1)
sol_analytic_rec_lower = interpolate(g_1, V_2)

model_problem = problem_def.ModelProblem(f_1)
schwarz_algorithm = cm.SchwarzMethodAlgebraic(V_1, rec_upper, bm_1, model_problem, g_1,
                                              V_2, rec_lower, bm_2, model_problem, g_1)
u1, u2 = schwarz_algorithm.solve(1e-6, 100)
print(f"Error of u1: {errornorm(sol_analytic_rec_upper, u1,'L2', mesh=rec_upper)}")
print(f"Error of u2: {errornorm(sol_analytic_rec_lower, u2,'L2', mesh=rec_lower)}")
height_1 = mid_intersection + delta / 2 - p0[1]
height_2 = height - height_1 + delta
omega_1 = {"x": (p0[0], p0[0]+length), "y": (p0[1], p0[1]+height_1)}
omega_2 = {"x": (p0[0], p0[0]+length), "y": (mid_intersection - delta / 2, mid_intersection - delta / 2 + height_2)}
overlap_indices_in_V1 = analyser.get_overlap_dof_indices(V_1, omega_1, omega_2)
print(f"Found {len(overlap_indices_in_V1)} DoFs from mesh1 in the overlap region.")
error, std = analyser.compute_dof_interface_error(u1, u2, overlap_indices_in_V1)
print(error)
print(std)
vis_func = Function(V_1)
vis_func.vector()[overlap_indices_in_V1] = 1
File("output_files/vis_func.pvd") << vis_func
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