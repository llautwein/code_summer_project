import matplotlib.pyplot as plt
from dolfin import *
import pandas as pd
import seaborn as sns
import numpy as np

class Visualiser:
    """
    A class summarising all plot routines.
    """

    def __init__(self):
        plt.rcParams.update({
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14
        })

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
        sns.set_theme(style="whitegrid", context="talk")
        for d, data in results.items():
            h = data['h']
            L2 = data['L2']
            plt.loglog(h, L2, '-o', label=f'L2-error for degree {d}')
        for d, data in results.items():
            h = data['h']
            H1 = data['H1']
            plt.loglog(h, H1, '-x', label=f'H1-error for degree {d}')
        plt.xlabel('Mesh element size h')
        plt.ylabel('Error')
        plt.xlim(0.01, 1)
        plt.legend()
        plt.savefig("saved_figures/convergence_verification.png", dpi=500)

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
        #plt.savefig("saved_figures/mesh_lower.png", dpi=500)
        #plt.close()

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

        sns.set_theme(style="whitegrid", context="talk")

        # Plot 1: Total Time vs. DoFs
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Total DoFs', y='Time (s)', hue='Method', marker='o', style='Method')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Total Degrees of Freedom (DoFs)')
        plt.ylabel('Overall Run Time (s)')
        plt.legend(title='Method')
        plt.grid(True, which="both", ls="--")
        plt.savefig("saved_figures/ddm_comp_totaldofs_time.png", dpi=500)
        plt.close()

        # Plot 2: Iterations vs. DoFs
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Total DoFs', y='Iterations', hue='Method', marker='o', style='Method')
        plt.xscale('log')
        plt.xlabel('Total Degrees of Freedom (DoFs)')
        plt.ylabel('Iterations')
        plt.legend(title='Method')
        plt.grid(True, which="both", ls="--")
        plt.savefig("saved_figures/ddm_comp_totaldofs_iterations.png", dpi=500)
        plt.close()

        # Plot 3: Iterations vs. Interface DoFS
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Interface DoFs', y='Iterations', hue='Method', marker='o', style='Method')
        plt.xscale('log')
        plt.xlabel('Interface Degrees of Freedom (DoFs)')
        plt.ylabel('Iterations')
        plt.legend(title='Method')
        plt.grid(True, which="both", ls="--")
        plt.savefig("saved_figures/ddm_comp_interfacedofs_iterations.png", dpi=500)
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="Total DoFs", y='Error_1', hue='Method', marker='o', style='Method')
        plt.xscale('log')
        plt.xlabel('Total Degrees of Freedom (DoFs)')
        plt.ylabel('Error')
        plt.yscale('log')
        plt.legend(title='Method')
        plt.grid(True, which="both", ls="--")
        plt.savefig("saved_figures/ddm_comp_totaldofs_error1.png", dpi=500)
        plt.close()

    def analyse_algebraic_schwarz_plot(self, results_path, x_axis, y_axis, fixed_params_dict, compare_by):
        df = pd.read_csv(results_path)
        print(results_path)
        sns.set_theme(style="whitegrid")

        df_copy = df.copy()
        # Filter the specified subset of the data to plot
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
        # plot the raw data
        sns.lineplot(data=df_copy, x=x_axis, y=y_axis,
                     hue=compare_by, ax=ax, marker='o')

        # Creating the labels manually
        legend_handles = []
        legend_labels = []
        i = 0
        # Plot the fitted data and create custom legend
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
                                  label=f'Fit (m≈{slope:.3f})', color=palette[i])
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

    def iterations_delta_scenarios_plot(self, scenarios: dict, x_axis: str, y_axis: str, fixed_params:dict = None):
        """
        Compares multiple experimental mesh scenarios by creating a plot from different csv files.
        :param scenarios: A dictionary of mesh scenarios.
        :param x_axis: Column name of the x-axis.
        :param y_axis: Column name of the y-axis.
        :param fixed_params: Dictionary of fixed parameters.
        """
        all_dfs = []
        for name, path in scenarios.items():
            try:
                df = pd.read_csv(path)
                df['Scenario'] = name
                all_dfs.append(df)
            except FileNotFoundError as e:
                print(f"Error: Could not find a results file. {e}")
                return

        df_filtered = pd.concat(all_dfs, ignore_index=True)

        if fixed_params is not None:
            for param, value in fixed_params.items():
                if pd.api.types.is_float_dtype(df_filtered[param]):
                    mask = pd.Series(False, index=df_filtered.index)
                    for v in value:
                        mask |= np.isclose(df_filtered[param], v)
                    df_filtered = df_filtered[mask]
                else:
                    df_filtered = df_filtered[df_filtered[param].isin(value)]

            if df_filtered.empty:
                print("Warning: No data matches the specified fixed parameters. Nothing to plot.")
                return

        sns.set_theme(style="whitegrid", context="talk")
        plt.figure(figsize=(10, 7))
        ax = plt.gca()  # Get the current axes to plot on

        # Get the color palette that Seaborn just used for the raw data
        palette = sns.color_palette(n_colors=len(df_filtered['Scenario'].unique()))

        # Plot the raw data using Seaborn. This will set up colors and the initial legend.
        ax = sns.lineplot(
            data=df_filtered,
            x=x_axis,
            y=y_axis,
            hue='Scenario',
            marker='o',
            linestyle='-',
            legend="full",
            palette=palette,
            ax=ax
        )

        for i, (name, group) in enumerate(df_filtered.groupby(['Scenario', "Polynomial Degree d"])):
            group = group.sort_values(by='Interface Width')

            slope = group['Fit Slope'].iloc[0]
            name = ("conforming", "independent")

            ax.plot(group['Interface Width'], group['Fit Iterations'],
                    linestyle='--',
                    label=rf"Fit for {name[i]} ($\delta^{{-{slope:.3f}}}$)",
                    color=palette[i])


        plt.xlabel(r"Interface Width ($\delta$)")
        plt.xscale('log')
        plt.yscale('log')

        plt.legend(title='Mesh Scenario')

        plt.grid(True, which="both", ls="--")
        plt.savefig("saved_figures/delta_dependence_comparison_gmsh.png", dpi=500)
        plt.close()
        #plt.show()

    @staticmethod
    def plot_parameter_study(results_path: str, x_axis: str, y_axis: str, hue: str,
                             fixed_params: dict=None, plot_fit: bool=False, x_log: bool=False, y_log: bool=False,
                             save_fig: bool=False, fig_name: str="results_fig", dpi: int=500)->None:
        """
        Creates a flexible lineplot of the provided data.
        :param results_path: Path to the csv results file.
        :param x_axis: Column name of the x-axis.
        :param y_axis: Column name of the y-axis.
        :param hue: Column to use for colouring the lines.
        :param fixed_params: A dictionary of fixed parameters.
        :param plot_fit: If True, plots the fitted line from the csv (delta^(-m))
        """
        try:
            df = pd.read_csv(results_path)
        except FileNotFoundError:
            print(f"Error: Could not find a results file at {results_path}")
            return

        df_filtered = df.copy()
        if fixed_params is not None:
            for param, value in fixed_params.items():
                if pd.api.types.is_float_dtype(df_filtered[param]):
                    mask = pd.Series(False, index=df_filtered.index)
                    for v in value:
                        mask |= np.isclose(df_filtered[param], v)
                    df_filtered = df_filtered[mask]
                else:
                    df_filtered = df_filtered[df_filtered[param].isin(value)]

        if df_filtered.empty:
            print("Warning: No data matches the specified fixed parameters. Nothing to plot.")
            return

        #if hue=="Offset Percentage":
        #df_filtered["Offset Percentage"] *= 100


        sns.set_theme(style="whitegrid", context="talk")
        hue_values = df_filtered[hue].unique()
        palette = sns.color_palette(n_colors=len(hue_values))
        plt.figure(figsize=(10, 7))
        ax = sns.lineplot(
            data=df_filtered,
            x=x_axis,
            y=y_axis,
            errorbar=("sd", 1),
            hue=hue,
            marker="o",
            legend="full",
            palette=palette
        )

        if plot_fit:
            for i, val in enumerate(hue_values):
                group = df_filtered[df_filtered[hue] == val].sort_values(by=x_axis)
                slope = group['Fit Slope'].iloc[0]
                ax.plot(
                    group[x_axis],
                    group[f"Fit {y_axis}"],
                    linestyle='--',
                    color=palette[i],
                    label=rf'Fit with $\delta_1^{{{-slope:.3f}}}$'
                )

        if x_log:
            plt.xscale('log')
        if y_log:
            plt.yscale('log')
        plt.legend(title=hue)
        plt.tight_layout()
        plt.xlabel(r"Interface Width $\delta_1$")


        if save_fig:
            path = "saved_figures/" + fig_name
            plt.savefig(
                path,
                dpi=dpi
            )
            plt.close()
        else:
            plt.show()






