from HelperModules import geometry_parser, visualiser
from dolfin import *
import matplotlib.pyplot as plt

geo_parser = geometry_parser.GeometryParser(0.2)
geo_parser.rectangle_mesh((0, 0), 1, 1, "rectangle")
geo_parser.circle_mesh((1, 1), 0.5, "circle")
mesh_rectangle = geo_parser.load_mesh("rectangle")
mesh_circle = geo_parser.load_mesh("circle")

vs = visualiser.Visualiser()
vs.mesh_plot([mesh_rectangle, mesh_circle], True)