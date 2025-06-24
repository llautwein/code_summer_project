from HelperModules import geometry_parser, visualiser
from dolfin import *
import FemSolver as fem_solver
import ProblemDefinition as problem_def
import matplotlib.pyplot as plt
import CompositionMethod as cm
import numpy as np

geo_parser = geometry_parser.GeometryParser(0.1)
geo_parser.rectangle_mesh((0, 0), 1, 1, "rectangle")
geo_parser.circle_mesh((1, 1), 0.5, "circle")
mesh_rectangle = geo_parser.load_mesh("rectangle")
mesh_circle = geo_parser.load_mesh("circle")
#mesh_rectangle = UnitSquareMesh(4, 4)

boundary_markers_1 = MeshFunction("size_t", mesh_rectangle, 1)
boundary_markers_1.set_all(0)
boundary_markers_2 = MeshFunction("size_t", mesh_circle, 1)
boundary_markers_2.set_all(0)

class Gamma_1(SubDomain):
    def __init__(self, cx, cy, r):
        super().__init__()
        self.cx = cx
        self.cy = cy
        self.r = r

    def inside(self, x, on_boundary):
        return ((x[0] - self.cx)**2 + (x[1] - self.cy)**2 < self.r**2) and on_boundary

class Boundary_Omega(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class Gamma_2(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] < 1 and x[1] < 1) and on_boundary


boundary_omega = Boundary_Omega()
boundary_omega.mark(boundary_markers_1, 1)
boundary_omega.mark(boundary_markers_2, 1)
gamma_1 = Gamma_1(1, 1, 0.5)
gamma_1.mark(boundary_markers_1, 2)
gamma_2 = Gamma_2()
gamma_2.mark(boundary_markers_2, 2)
File("boundary_markers.pvd") << boundary_markers_1
File("boundary_markers.pvd") << boundary_markers_2

V_1 = FunctionSpace(mesh_rectangle, "CG", 1)
g_1 = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=6)
f_1 = Expression("(8*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2)

V_2 = FunctionSpace(mesh_circle, "CG", 1)
g_2 = Expression("1 + x[0]*x[0] + 2 * x[1]*x[1]", degree=2)
f_2 = Constant(-6.0)

helmholtz = problem_def.HelmholtzProblem(f_1)
poisson = problem_def.PoissonProblem(f_2)


schwarz_algorithm = cm.SchwarzMethod_primitive(V_1, mesh_rectangle, boundary_markers_1, helmholtz, g_1,
                                               V_2, mesh_circle, boundary_markers_2, helmholtz, g_1)
#u1, u2 = schwarz_algorithm.solve(1e-7, 100)

"""
solver = fem_solver.FemSolver()
V = FunctionSpace(mesh, "CG", 1)
u0 = Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=6)
f = Expression("(8*pi*pi+1)*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2)
helmholtz = problem_def.HelmholtzProblem(f, u0)
"""
vs = visualiser.Visualiser()
#vs.mesh_plot([mesh_rectangle, mesh_circle], True)
#vs.heatmap_plot(u1, mesh_rectangle)
#vs.heatmap_plot(u2, mesh_circle)
#vs.plot_overlap_difference(u1, mesh_rectangle, u2)

