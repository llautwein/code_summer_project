import matplotlib.pyplot as plt
from dolfin import *
import pandas as pd
import seaborn as sns
import numpy as np

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

    def compare_ddm_methods_plot(self, results_path):
        df = pd.read_csv(results_path)

        sns.set_theme(style="whitegrid")

        # Plot 1: Total Time vs. DoFs
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Total DoFs', y='Time (s)', hue='Method', marker='o', style='Method')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Total Degrees of Freedom (DoFs)')
        plt.ylabel('Overall Run Time (s)')
        plt.legend(title='Method')
        plt.grid(True, which="both", ls="--")
        plt.show()

        # Plot 2: Iterations vs. DoFs
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Total DoFs', y='Iterations', hue='Method', marker='o', style='Method')
        plt.xscale('log')
        plt.xlabel('Total Degrees of Freedom (DoFs)')
        plt.ylabel('Number of Iterations')
        plt.legend(title='Method')
        plt.grid(True, which="both", ls="--")
        plt.show()

    def analyse_algebraic_schwarz_plot(self, results_path, x_axis, y_axis, fixed_params_dict, compare_by):
        df = pd.read_csv(results_path)
        sns.set_theme(style="whitegrid")

        df_copy = df.copy()
        for param_name, list_values in fixed_params_dict.items():
            if pd.api.types.is_float_dtype(df_copy[param_name]):
                mask = pd.Series(False, index=df_copy.index)
                for val in list_values:
                    mask |= np.isclose(df_copy[param_name], val)
                df_copy = df_copy[mask]
            else:
                df_copy = df_copy[df_copy[param_name].isin(list_values)]
        plt.figure()
        ax = plt.gca()
        sns.lineplot(data=df_copy, x=x_axis, y=y_axis,
                     hue=compare_by, ax=ax, marker='o')

        # Creating the labels manually
        legend_handles = []
        legend_labels = []
        i = 0

        groups = df_copy.groupby(compare_by)
        palette = sns.color_palette(n_colors=len(groups))
        for name, group in groups:
            # Sort the data to ensure the line is drawn correctly
            group = group.sort_values(by=x_axis)

            # Get the pre-calculated slope from the DataFrame
            slope = group['Fit Slope'].iloc[0]

            handle_raw, = ax.plot([], [], marker='o', linestyle='-', color=palette[i],
                                  label=f'Data ({compare_by}={name})')
            legend_handles.append(handle_raw)

            # Plot the fitted data from the 'Fit Iterations' column
            handle_fit, = ax.plot(group[x_axis], group["Fit Iterations"],
                                  linestyle='--',
                                  label=f'Fit (m≈{slope:.3f})')
            legend_handles.append(handle_fit)
            legend_labels.append(f'Iterations for d={name}')
            legend_labels.append(rf'Fit with $\delta^{{-{slope:.3f}}}$')
            i+=1

        plt.xlabel(x_axis)
        plt.xscale('log')
        plt.ylabel(y_axis)
        plt.yscale('log')
        plt.legend(handles=legend_handles, labels=legend_labels, title=compare_by)
        plt.show()




