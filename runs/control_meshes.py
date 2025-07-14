from HelperModules import GeometryParser as gp
import InterfaceHandler as ih
from HelperModules import visualiser as vis
import numpy as np
from dolfin import *
import fenics


h = np.logspace(-0.5, -2.5, 30)[5]
left_bottom_corner_1 = (0, 0)
length_1 = 1
height_1 = 0.55
left_bottom_corner_2 = (0, 0.45)
length_2 = 1
height_2 = 0.55

#geo_parser = gp.GeometryParser(h)
#geo_parser.rectangle_mesh(left_bottom_corner_1, length_1, height_1, "rectangle_lower")
mesh_rectangle_1 = RectangleMesh(Point(0, 0.45), Point(1, 1), 10, 10)
#geo_parser.rectangle_mesh(left_bottom_corner_2, length_2, height_2, "rectangle_upper", "triangle")
#mesh_rectangle_1 = geo_parser.load_mesh("rectangle_upper")
mesh_rectangle_2 = RectangleMesh(Point(0, 0), Point(1, 0.55), 10, 10)

interface_handler = ih.OverlappingRectanglesInterfaceHandler(mesh_rectangle_1, mesh_rectangle_2)

boundary_markers_1, boundary_markers_2 = interface_handler.mark_interface_boundaries(left_bottom_corner_2[1], height_1)

File("output_files/boundary_markers.pvd") << boundary_markers_2

visualiser = vis.Visualiser()
visualiser.mesh_plot([mesh_rectangle_1, mesh_rectangle_2])
