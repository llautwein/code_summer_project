import os
import meshio
from dolfin import Point, RectangleMesh, Mesh, MeshFunction, XDMFFile, SubMesh, SubDomain, DOLFIN_EPS
import fenics
from typing import *
import numpy as np

class GeometryParser:
    """
    This helper class generates the .geo-file to obtain meshes for basic geometries.
    """
    def __init__(self):
        self.points_count = 0
        self.line_circs_count = 0
        self.curve_loop_count = 0
        self.physical_curve_count = 0
        self.plane_surface_count = 0
        self.physical_surface_count = 0

    def initialize_file(self, file_name):
        with open(f"meshes/{file_name}.geo", "w") as file:
            file.write("//parsed geo-file\n")

    def define_variables(self, file_name, lc):
        with open(f"meshes/{file_name}.geo", "a") as file:
            file.write("\n")
            file.write(f"lc = {lc};\n")

    def point(self, x, y, lc_str="lc"):
        self.points_count += 1
        output = (
            f"Point({self.points_count}) = "
            "{"
            f"{x}, {y}, 0, {lc_str}"
            "};\n"
        )
        return output

    def line(self, start, end):
        self.line_circs_count += 1
        output = (
            f"Line({self.line_circs_count}) = "
            "{"
            f"{start}, {end}"
            "};\n"
        )
        return output

    def curve_loop(self, start, end):
        self.curve_loop_count += 1
        output = f"Curve Loop({self.curve_loop_count}) = "
        output += "{"
        for i in range(start, end):
            output += f"{i},"
        output += f"{end}"
        output += "};\n"
        return output

    def curve_loop_from_list(self, idcs):
        self.curve_loop_count += 1
        output = f"Curve Loop({self.curve_loop_count}) = "
        output += "{"
        for i in range(len(idcs)-1):
            output += f"{idcs[i]},"
        output += f"{idcs[-1]}"
        output += "};\n"
        return output

    def circle(self, first_point, center, second_point):
        self.line_circs_count += 1
        output = (
            f"Circle({self.line_circs_count}) = "
            "{"
            f"{first_point}, {center}, {second_point}"
            "};\n"
        )
        return output

    def physical_curve(self, idcs):
        self.physical_curve_count += 1
        output = (
            f"Physical Curve({self.physical_curve_count}) = "
            "{"
        )
        for i in range(len(idcs) - 1):
            output += f"{idcs[i]},"
        output += f"{idcs[len(idcs) - 1]}"
        output += "};\n"
        return output

    def plane_surface(self, index):
        self.plane_surface_count += 1
        output = f"Plane Surface({self.plane_surface_count}) ="
        output += "{"
        output += f"{index}"
        output += "};\n"
        return output

    def physical_surface(self, index):
        self.physical_surface_count += 1
        output = f"Physical Surface({self.plane_surface_count}) ="
        output += "{"
        output += f"{index}"
        output += "};\n"
        return output

    def gradual_mesh_option(self, file_name, interface_edge_idx, lc_min, lc_max, dist_min, dist_max):
        with open(f"meshes/{file_name}.geo", "a") as file:
            file.write("\n")
            file.write("Field[1] = Distance;\n")
            file.write("Field[1].NNodesByEdge = 100;\n")
            file.write(f"Field[1].EdgesList = {{{interface_edge_idx}}};\n")

            file.write("Field[2] = Threshold;\n")
            file.write("Field[2].IField = 1;\n")
            file.write(f"Field[2].LcMin = {lc_min};\n")
            file.write(f"Field[2].LcMax = {lc_max};\n")
            file.write(f"Field[2].DistMin = {dist_min};\n")
            file.write(f"Field[2].DistMax = {dist_max};\n")
            file.write("Mesh.MeshSizeExtendFromBoundary = 0;")
            file.write("\n")
            file.write("Background Field = 2;")


    def convert_to_xdmf(self, file_name, cell_type="triangle"):
        # Create the corresponding paths
        msh_path = f"meshes/{file_name}.msh"
        xdmf_path = f"meshes/{file_name}.xdmf"

        # try reading the msh-file
        try:
            msh = meshio.read(f"meshes/{file_name}.msh")
        except Exception as e:
            print(f"Error reading {msh_path}: {e}")
            return

        # Filter for the correct cells and save them for writing in the xdmf-file
        target_meshio_type = 'quad' if cell_type == 'quad' else 'triangle'

        cells_to_write = None
        for cell_block in msh.cells:
            if cell_block.type == target_meshio_type:
                cells_to_write = cell_block
                break

        if cells_to_write is None:
            print(f"Error: Could not find cells of type '{target_meshio_type}' in {msh_path}.")
            return

        out_mesh = meshio.Mesh(
            points=msh.points,
            cells=[cells_to_write]
        )
        meshio.write(
            xdmf_path,
            out_mesh,
            file_format="xdmf"
        )
        print("Successfully converted mesh to xdmf-format")

    def load_mesh(self, file_name):
        mesh = fenics.Mesh()
        with fenics.XDMFFile(f"meshes/{file_name}.xdmf") as xdmf_file:
            xdmf_file.read(mesh)
        return mesh

    def rectangle_mesh(self, lc, left_bottom_corner, length, height, file_name,
                       refine_edge=None, refinement_factor=10.0, transition_ratio=0.25):
        """
        Generates a rectangular mesh, with an option to refine one edge.
        """
        self.points_count = 0
        self.line_circs_count = 0
        self.curve_loop_count = 0
        self.plane_surface_count = 0
        self.physical_surface_count = 0

        self.initialize_file(file_name)
        self.define_variables(file_name, lc)

        commands = []
        x0 = left_bottom_corner[0]
        y0 = left_bottom_corner[1]

        # Points
        p1_idx, p2_idx, p3_idx, p4_idx = 1, 2, 3, 4
        commands.append(self.point(x0, y0))
        commands.append(self.point(x0 + length, y0))
        commands.append(self.point(x0 + length, y0 + height))
        commands.append(self.point(x0, y0 + height))

        # Lines (with correct identification)
        l_bottom_idx = self.line_circs_count + 1; commands.append(self.line(p1_idx, p2_idx))
        l_right_idx = self.line_circs_count + 1;  commands.append(self.line(p2_idx, p3_idx))
        l_top_idx = self.line_circs_count + 1; commands.append(self.line(p3_idx, p4_idx))
        l_left_idx = self.line_circs_count + 1; commands.append(self.line(p4_idx, p1_idx))

        # Surface
        commands.append(self.curve_loop(l_bottom_idx, l_left_idx))
        commands.append(self.plane_surface(self.curve_loop_count))

        if refine_edge is not None:
            edge_map = {'bottom': l_bottom_idx, 'right': l_right_idx, 'top': l_top_idx, 'left': l_left_idx}
            if refine_edge not in edge_map:
                raise ValueError(f"Invalid refine_edge. Choose from {list(edge_map.keys())}")

            interface_line_idx = "{" + str(edge_map[refine_edge]) + "}"
            lc_min = lc / refinement_factor
            dist_max = min(length, height) * transition_ratio

            num_nodes_on_edge = int(round(length / lc_min))

            commands.append(f"Transfinite Line {{{edge_map[refine_edge]}}} = {num_nodes_on_edge + 1};")

            commands.append("\n// --- Mesh Refinement Fields ---")
            commands.append("Field[1] = Distance;")
            commands.append(f"Field[1].EdgesList = {interface_line_idx};")
            commands.append("Field[2] = Threshold;")
            commands.append("Field[2].IField = 1;")
            commands.append(f"Field[2].LcMin = {lc_min};")
            commands.append(f"Field[2].LcMax = {lc};")
            commands.append(f"Field[2].DistMin = 0.0;")
            commands.append(f"Field[2].DistMax = {dist_max};")

            # This is the robust way to combine the field with the boundary lc
            commands.append("Field[3] = Min;")
            commands.append("Field[3].FieldsList = {2};")
            commands.append("Background Field = 3;")

        # 4. Final commands
        commands.append("Mesh 2;")
        commands.append(self.physical_surface(self.plane_surface_count))

        # 5. Write all commands to the file at once
        with open(f"meshes/{file_name}.geo", "a") as file:
            file.write("\n".join(commands))

        # 6. Run Gmsh and Convert
        os.system(f"gmsh meshes/{file_name}.geo -2 -format msh2 -o meshes/{file_name}.msh")
        self.convert_to_xdmf(file_name)

    def circle_mesh(self, lc, center, radius, file_name):
        self.initialize_file(file_name)
        self.define_variables(file_name, lc)
        with open(f"meshes/{file_name}.geo", "a") as file:
            file.write("\n// Circle Geometry\n")
            file.write(self.point(center[0] + radius, center[1]))
            file.write(self.point(center[0], center[1] + radius))
            file.write(self.point(center[0] - radius, center[1]))
            file.write(self.point(center[0], center[1] - radius))
            file.write(self.point(center[0], center[1]))
            file.write("\n")
            file.write(self.circle(self.points_count - 4, self.points_count, self.points_count - 3))
            file.write(self.circle(self.points_count - 3, self.points_count, self.points_count - 2))
            file.write(self.circle(self.points_count - 2, self.points_count, self.points_count - 1))
            file.write(self.circle(self.points_count - 1, self.points_count, self.points_count - 4))
            file.write("\n")
            file.write(self.curve_loop(self.line_circs_count - 3, self.line_circs_count))
            file.write("\n")
            file.write("\n")
            file.write(self.plane_surface(self.curve_loop_count))
            file.write("Mesh 2;\n")
            file.write(self.physical_surface(self.plane_surface_count))

        os.system(f"gmsh meshes/{file_name}.geo -2")
        self.convert_to_xdmf(file_name)

    def create_independent_meshes(self, p0: Point, length: float, height: float,
                                  mid_intersection: float, delta: float, h: float,
                                  mesh_option: str="built-in", gmsh_parameters: dict=None)->Tuple[Mesh, Mesh]:
        """
        Creates two independent, rectangular meshes, which overlap in a strip of width delta. In the overlap,
        both meshes will cut through elements of the other mesh respectively.
        :param p0: The left bottom corner of the lower rectangle.
        :param length: The horizontal length of both rectangles.
        :param height: The combined, overall height of both rectangles.
        :param mid_intersection: The y-coordinate of the centre of the overlap.
        :param delta: The interface width.
        :param h: The mesh resolution in terms of the element size.
        :return: The mesh of the upper and the lower rectangle.
        """
        # Calculate the heights of the overlapping rectangles
        height_1 = mid_intersection + delta / 2 - p0[1]
        height_2 = height - height_1 + delta

        # Lower rectangle:
        # The point p0_lower is the left bottom corner, p1_lower the right top corner of the lower rectangle.
        p0_lower = Point(p0[0], p0[1])
        p1_lower = Point(p0[0] + length, p0[1] + height_1)

        # Upper rectangle (same principle):
        # p0_upper is the left bottom corner, p1_upper the top right corner of the upper rectangle
        p0_upper = Point(p0[0], mid_intersection - delta / 2)
        p1_upper = Point(p0[0] + length, mid_intersection - delta / 2 + height_2)

        if mesh_option=="built-in":
            # Define the mesh resolution in the horizontal and vertical directions based on the element size
            nx = max(1, int(round(length / h)))
            ny_1 = max(1, int(round(height_1 / h)))
            ny_2 = max(1, int(round(height_2 / h)))

            # Create meshes directly using FEniCS
            rectangle_upper = RectangleMesh(p0_upper, p1_upper, nx, ny_1)
            rectangle_lower = RectangleMesh(p0_lower, p1_lower, nx, ny_2)
        elif mesh_option=="gmsh":
            if gmsh_parameters is None:
                gmsh_parameters = {}

            refinement_factor = gmsh_parameters.get('refinement_factor', 10.0)
            transition_ratio = gmsh_parameters.get('transition_ratio', 0.25)

            refine_lower_edge = None
            refine_upper_edge = None
            if gmsh_parameters.get('refine_at_interface', False):
                refine_lower_edge = 'top'
                refine_upper_edge = 'bottom'

            self.rectangle_mesh(
                lc=h,
                left_bottom_corner=p0_lower,
                length=length,
                height=height_1,
                file_name="rectangle_lower",
                refine_edge=refine_lower_edge,
                refinement_factor=refinement_factor,
                transition_ratio=transition_ratio
            )
            rectangle_lower = self.load_mesh("rectangle_lower")

            self.rectangle_mesh(
                lc=h,
                left_bottom_corner=p0_upper,
                length=length,
                height=height_2,
                file_name="rectangle_upper",
                refine_edge=refine_upper_edge,
                refinement_factor=refinement_factor,
                transition_ratio=transition_ratio
            )
            rectangle_upper = self.load_mesh("rectangle_upper")
        else:
            raise ValueError(f"Unknown mesh option '{mesh_option}'. Choose 'built-in' or 'gmsh'.")

        return rectangle_upper, rectangle_lower



    def create_conforming_meshes(self, p0: Point, length: float, height: float,
                                 mid_intersection: float, delta: float, N_overlap: int=1,
                                 mesh_option: str="built-in", gmsh_parameters: dict=None)->Tuple[Mesh, Mesh]:
        """
        Helper method that creates two rectangular meshes with an interface of width delta and perfectly matching
        nodes within that overlap. The mesh size is computed from the number of element layers within the
        overlap. When decreasing the overlap, the mesh element size necessarily decrease as well.
        :param p0: The bottom left corner of the global domain.
        :param length: Total length of the global domain.
        :param height: Total height of the global domain.
        :param mid_intersection: The y-coordinate of the centre of the overlap.
        :param delta: The width of the overlap.
        :param N_overlap: The number of element layers within the overlap
        :return: The mesh of the upper and the lower rectangle.
        """
        print(f"Creating conforming meshes with delta={delta:.4f}")

        # Define key y-coordinates
        x0 = p0[0]
        y_bottom = p0[1]
        y_interface_lower = mid_intersection - delta / 2
        y_interface_upper = mid_intersection + delta / 2
        y_top = p0[1] + height

        # Define heights of the three distinct regions
        h_lower_region = y_interface_lower - y_bottom
        h_upper_region = y_top - y_interface_upper

        ny_overlap = max(1, N_overlap)

        if mesh_option=="built-in":
            # Calculate number of cell divisions for each region
            uniform_cell_size = delta / ny_overlap

            # Define the mesh resolution in the horizontal and vertical directions based on the element size
            nx = int(round(length / uniform_cell_size))
            ny_lower = int(round(h_lower_region / uniform_cell_size))
            ny_upper = int(round(h_upper_region / uniform_cell_size))
            ny_total = ny_lower + ny_overlap + ny_upper

            # Adjust the heights slightly in order to fit in the required number of cells.
            h_lower_actual = ny_lower * uniform_cell_size
            h_upper_actual = ny_upper * uniform_cell_size
            y_bottom_actual = y_interface_lower - h_lower_actual
            y_top_actual = y_interface_upper + h_upper_actual
            print(f"Original y_bottom: {p0[1]:.4f}, Adjusted y_bottom: {y_bottom_actual:.4f}")
            print(f"Original y_top: {p0[1] + height:.4f}, Adjusted y_top: {y_top_actual:.4f}")

            # Create the single, global mesh
            p0_global = Point(p0[0], y_bottom_actual)
            p1_global = Point(p0[0] + length, y_top_actual)
            global_mesh = RectangleMesh(p0_global, p1_global, nx, ny_total)

            # Define subdomains based on y-coordinates
            class LowerSubdomain(SubDomain):
                def inside(self, x, on_boundary):
                    return x[1] <= y_interface_upper + DOLFIN_EPS

            class UpperSubdomain(SubDomain):
                def inside(self, x, on_boundary):
                    return x[1] >= y_interface_lower - DOLFIN_EPS

            # Extract the submeshes
            mesh_lower = SubMesh(global_mesh, LowerSubdomain())
            mesh_upper = SubMesh(global_mesh, UpperSubdomain())
        elif mesh_option=="gmsh":
            self.points_count = 0
            self.line_circs_count = 0
            self.curve_loop_count = 0
            self.plane_surface_count = 0
            self.physical_surface_count = 0

            lc_coarse = gmsh_parameters.get('lc_coarse', 0.1)
            lc_overlap = delta / N_overlap
            file_name = "conforming_mesh"
            self.initialize_file(file_name)
            self.define_variables(file_name, lc_coarse)

            commands = []
            commands.append(f"lc_overlap = {lc_overlap};\n")
            # Define all points
            commands.append(self.point(x0, y_bottom))
            commands.append(self.point(x0 + length, y_bottom))
            commands.append(self.point(x0 + length, y_interface_lower, "lc_overlap"))
            commands.append(self.point(x0, y_interface_lower, "lc_overlap"))
            commands.append(self.point(x0, y_interface_upper, "lc_overlap"))
            commands.append(self.point(x0 + length, y_interface_upper, "lc_overlap"))
            commands.append(self.point(x0 + length, y_top))
            commands.append(self.point(x0, y_top))

            # Define all lines
            commands.append(self.line(1, 2)) # bottom
            commands.append(self.line(2, 3)) # right lower
            commands.append(self.line(3, 4)) # interface lower boundary
            commands.append(self.line(4, 1)) # left lower
            commands.append(self.line(3, 6)) # right interface
            commands.append(self.line(4, 5)) # left interface
            commands.append(self.line(5, 6)) # interface upper boundary
            commands.append(self.line(6, 7)) # right upper
            commands.append(self.line(7, 8)) # top
            commands.append(self.line(8, 5)) # left upper

            # lower rectangle with tag 1
            commands.append(self.curve_loop_from_list([1, 2, 3, 4]))
            commands.append(self.plane_surface(self.curve_loop_count))
            commands.append(self.physical_surface(self.plane_surface_count))

            # upper rectangle with tag 2
            commands.append(self.curve_loop_from_list([7, 8, 9, 10]))
            commands.append(self.plane_surface(self.curve_loop_count))
            commands.append(self.physical_surface(self.plane_surface_count))

            # overlap with tag 3
            commands.append(self.curve_loop_from_list([-3, 5, -7, -6]))
            commands.append(self.plane_surface(self.curve_loop_count))
            commands.append(self.physical_surface(self.plane_surface_count))

            commands.append("Mesh 2;")

            with open(f"meshes/{file_name}.geo", "a") as file:
                file.write("\n".join(commands))

            os.system(f"gmsh meshes/{file_name}.geo -2 -format msh2 -o meshes/{file_name}.msh")

            # Post-processing of the global mesh to split it into the upper and lower subdomain
            # First, read the complete mesh file, which has tags 1 (lower), 2 (upper) and 3 (interface)
            msh_path = f"meshes/{file_name}.msh"
            print(f"Reading Gmsh output file: {msh_path}")
            msh = meshio.read(msh_path)
            target_meshio_type = 'triangle'

            try:
                # Extract all the cell coordinates and the respective tags
                cells_all = msh.get_cells_type(target_meshio_type)
                cell_tags = msh.get_cell_data("gmsh:physical", target_meshio_type)
            except KeyError:
                print(f"Error: Could not find cells of type '{target_meshio_type}' in {msh_path}.")
                return None, None

            # Mesh pruning function
            def prune_mesh(points_all, cells_subset):
                """
                Manual pruning function that takes a full list of mesh coordinates and subset of indices
                and returns the corresponding submesh coordinates.
                :param points_all: The full list of mesh coordinates.
                :param cells_subset: The list of indices of the desired subset of nodes.
                :return: The pruned mesh description.
                """
                # Find the unique vertex indices used by this subset of cells
                unique_vertex_indices = np.unique(cells_subset.flatten())

                # Create the new, pruned list of points
                points_pruned = points_all[unique_vertex_indices]

                # Since we now have less points in the mesh, we reindex them:
                # Create a map from the old, global vertex index to the new, local index
                old_to_new_map = np.full(points_all.shape[0], -1, dtype=np.int32)
                old_to_new_map[unique_vertex_indices] = np.arange(len(unique_vertex_indices))

                # Use the map to re-index the cell connectivity array
                cells_pruned = old_to_new_map[cells_subset]

                return points_pruned, cells_pruned

            # Create the lower subdomain
            # Create a boolean array (mask) that is 'True' for every cell whose tag is in [1, 3].
            mask_lower = np.isin(cell_tags, [1, 3])
            # Use the mask to select the desired cells, then prune the mesh.
            points_lower, cells_lower = prune_mesh(msh.points, cells_all[mask_lower])
            # Create a new meshio.Mesh object for the lower subdomain.
            meshio_lower = meshio.Mesh(points=points_lower, cells={target_meshio_type: cells_lower})

            # Create the upper subdomain
            # Similar process
            mask_upper = np.isin(cell_tags, [2, 3])
            points_upper, cells_upper = prune_mesh(msh.points, cells_all[mask_upper])
            meshio_upper = meshio.Mesh(points=points_upper, cells={target_meshio_type: cells_upper})

            # Write and Load Final Meshes
            meshio.write(f"meshes/{file_name}_lower.xdmf", meshio_lower)
            meshio.write(f"meshes/{file_name}_upper.xdmf", meshio_upper)

            mesh_lower = Mesh()
            with XDMFFile(f"meshes/{file_name}_lower.xdmf") as infile:
                infile.read(mesh_lower)

            mesh_upper = Mesh()
            with XDMFFile(f"meshes/{file_name}_upper.xdmf") as infile:
                infile.read(mesh_upper)

        else:
            raise ValueError(f"Unknown mesh option '{mesh_option}'. Choose 'built-in' or 'gmsh'.")

        return  mesh_upper, mesh_lower

    def create_circle_rectangle(self, midpoint_circle: Point, radius: float, length: float, height:float,
                                delta: float, h: float):
        """
        Creates a circular and a rectangular mesh which overlap (as given on the cover of Smith, Bjorstad, Gropp)
        :param p0: Left bottom corner of the rectangle.
        :param length: Length of the rectangle.
        :param height: Height of the rectangle.
        :param midpoint_circle: Midpoint of the circle.
        :param radius: Radius of the circle.
        :return: mesh_circle, mesh_rectangle
        """
        x0 = midpoint_circle[0] + radius - delta
        y0 = midpoint_circle[1] - height / 2
        p0 = Point(x0, y0)
        p1 = Point(x0 + length, y0 + height)
        nx = max(1, int(round(length / h)))
        ny = max(1, int(round(height / h)))
        mesh_rectangle = RectangleMesh(p0, p1, nx, ny)

        self.circle_mesh(h, midpoint_circle, radius, "mesh_circle")
        mesh_circle = self.load_mesh("mesh_circle")

        if (delta-radius)**2 + (height/2)**2 > radius**2:
            print("Warning: corners of the rectangle are not inside the circle.")

        return mesh_circle, mesh_rectangle






