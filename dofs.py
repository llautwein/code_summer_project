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

#boundary_markers_1 = MeshFunction("size_t", mesh_rectangle, 1)
#boundary_markers_1.set_all(0)
#boundary_markers_2 = MeshFunction("size_t", mesh_circle, 1)
#boundary_markers_2.set_all(0)

class Gamma_1(SubDomain):
    def __init__(self, cx, cy, r):
        super().__init__()
        self.cx = cx
        self.cy = cy
        self.r = r

    def inside(self, x, on_boundary):
        return ((x[0] - self.cx)**2 + (x[1] - self.cy)**2 < self.r**2 + DOLFIN_EPS) and on_boundary

class Boundary_Omega(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class Gamma_2(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] < 1 and x[1] < 1) and on_boundary


#boundary_omega = Boundary_Omega()
#boundary_omega.mark(boundary_markers_1, 1)
#boundary_omega.mark(boundary_markers_2, 1)
#gamma_1 = Gamma_1(1, 1, 0.5)
#gamma_1.mark(boundary_markers_1, 2)
#gamma_2 = Gamma_2()
#gamma_2.mark(boundary_markers_2, 2)

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

helmholtz = problem_def.HelmholtzProblem(f_1)
poisson = problem_def.PoissonProblem(f_2)

vs = visualiser.Visualiser()
vs.mesh_plot([mesh_rectangle, mesh_circle], True)

"""
# Algebraic version of the Schwarz method
schwarz_matrix = cm.SchwarzMethod_matrix(V_1, mesh_rectangle, boundary_markers_1, helmholtz, g_1,
                 V_2, mesh_circle, boundary_markers_2, helmholtz, g_1)
u1_m, u2_m = schwarz_matrix.solve(1e-4, 100)
"""
"""
# Alternating Schwarz method
schwarz_primitive = cm.SchwarzMethod_alternating(V_1, mesh_rectangle, boundary_markers_1, helmholtz, g_1,
                 V_2, mesh_circle, boundary_markers_2, helmholtz, g_1)
u1_p, u2_p = schwarz_primitive.solve(1e-4, 100)
"""
"""
# Algorithm that focuses on the interface problem
schwarz_operator = cm.SchwarzMethod_operator(V_1, mesh_rectangle, boundary_markers_1, helmholtz, g_1,
                                             V_2, mesh_circle, boundary_markers_2, poisson, g_2)
lambda_1, lambda_2, u1, u2 = schwarz_operator.solve(1e-4, max_iter=5)
"""

schwarz_operator_matrix = cm.SchwarzMethod_operator_matrix(V_1, mesh_rectangle, boundary_markers_1, helmholtz, g_1,
                                             V_2, mesh_circle, boundary_markers_2, helmholtz, g_1)
u1_om, u2_om = schwarz_operator_matrix.solve(1e-4, 50)
print("####################################################")
print("L2-error:")
print(errornorm(sol_analytic_rec, u1_om, "l2"))

vs.heatmap_plot(sol_analytic_rec, mesh_rectangle)
vs.heatmap_plot(sol_analytic_circ, mesh_circle)
vs.heatmap_plot(u1_om, mesh_rectangle)
vs.heatmap_plot(u2_om, mesh_circle)

