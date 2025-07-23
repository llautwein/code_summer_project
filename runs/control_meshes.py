from HelperModules import GeometryParser as gp
import InterfaceHandler as ih
from HelperModules import visualiser as vis
import numpy as np
from dolfin import *
import fenics

p0 = Point(0, -0.25)
length = 1
height = 2
mid_intersection = 0.75
delta = 0.1
h = 0.1
mesh_option = "gmsh"
gmsh_parameters = {"lc_coarse": 0.1}
visualiser = vis.Visualiser()
geo_parser = gp.GeometryParser()
rec_upper, rec_lower = geo_parser.create_conforming_meshes(p0, length, height, mid_intersection, delta,
                                     mesh_option=mesh_option, gmsh_parameters=gmsh_parameters)

interface_handler = ih.OverlappingRectanglesInterfaceHandler(rec_upper, rec_lower)
y_interface_of_upper_domain = mid_intersection - delta / 2
y_interface_of_lower_domain = mid_intersection + delta / 2
bm_1, bm_2 = interface_handler.mark_interface_boundaries(
    y_interface_of_upper_domain,
    y_interface_of_lower_domain
)
File("output_files/boundary_markers.pvd") << bm_2
visualiser.mesh_plot([rec_upper, rec_lower])
visualiser.mesh_plot([rec_upper])
visualiser.mesh_plot([rec_lower])
#interface_handler = ih.InterfaceHandler(mesh_circle, mesh_rectangle)
#bm_1, bm_2 = interface_handler.mark_interface_boundaries()
#File("output_files/boundary_markers.pvd") << bm_1
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