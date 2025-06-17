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

