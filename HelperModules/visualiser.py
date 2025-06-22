import matplotlib.pyplot as plt
from dolfin import *

class Visualiser:
    """
    A class summarising all plot routines.
    """
    @staticmethod
    def plot_3d(u_sol, mesh):
        vertex_coords = mesh.coordinates()
        x = vertex_coords[:, 0]
        y = vertex_coords[:, 1]

        z = u_sol.compute_vertex_values(mesh)

        triangles = mesh.cells()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_trisurf(x, y, triangles, z, cmap=plt.cm.viridis, linewidth=0.2)

        ax.set_title("3D Surface Plot of Solution")
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("u(x,y)")

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        plt.show()

    def convergence_rates_plot(self, results):
        for d, data in results.items():
            h = data['h']
            L2 = data['L2']
            plt.loglog(h, L2, '-o', label=f'L2-error for degree {d}')
        for d, data in results.items():
            h = data['h']
            H1 = data['H1']
            plt.loglog(h, H1, '-x', label=f'H1-error for degree {d}')
        plt.xlabel('Step size h')
        plt.ylabel('Error')
        plt.xlim(0.01, 1)
        plt.legend()
        plt.show()

    def mesh_plot(self, mesh_list, ax_equal=False):
        fig, ax = plt.subplots()

        for mesh in mesh_list:
            coords = mesh.coordinates()
            x = coords[:, 0]
            y = coords[:, 1]

            triangles = mesh.cells()

            ax.triplot(x, y, triangles, linewidth=0.5)

        if ax_equal:
            ax.set_aspect('equal')

        plt.show()

    @staticmethod
    def heatmap_plot(u_sol, mesh):

        fig, ax = plt.subplots(figsize=(8, 7))

        vertex_coords = mesh.coordinates()
        x = vertex_coords[:, 0]
        y = vertex_coords[:, 1]
        triangles = mesh.cells()
        z = u_sol.compute_vertex_values(mesh)


        contour = ax.tricontourf(x, y, triangles, z, levels=20, cmap=plt.cm.viridis)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")

        ax.set_aspect('equal')

        fig.colorbar(contour, ax=ax)

        plt.show()

    def plot_overlap_difference(self, u1, mesh1, u2):
        """
        Creates a heatmap of the difference |u1 - u2| in the overlap region.
        """
        print("\n--- Plotting difference function in overlap region ---")
        V1 = u1.function_space()
        u2_on_mesh1 = Function(V1)
        u2.set_allow_extrapolation(True)
        u2_on_mesh1.interpolate(u2)

        difference_func = project(abs(u1 - u2_on_mesh1), V1)

        self.heatmap_plot(difference_func, mesh1)

