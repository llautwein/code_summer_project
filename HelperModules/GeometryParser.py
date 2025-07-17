import os
import meshio
from dolfin import *
import fenics
from typing import *

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

    def point(self, x, y):
        self.points_count += 1
        output = (
            f"Point({self.points_count}) = "
            "{"
            f"{x}, {y}, 0, lc"
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
        print(f"Converting to {xdmf_path} for '{cells_to_write.type}' cells...")
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

    def rectangle_mesh(self, lc, left_bottom_corner, length, height, file_name, cell_type="triangle"):
        self.points_count = 0
        self.line_circs_count = 0
        self.curve_loop_count = 0
        self.plane_surface_count = 0
        self.physical_surface_count = 0

        self.initialize_file(file_name)
        self.define_variables(file_name, lc)
        with open(f"meshes/{file_name}.geo", "a") as file:
            # Specify the points and capture their indices
            x0, y0 = left_bottom_corner
            p1_idx = self.points_count + 1; file.write(self.point(x0, y0))
            p2_idx = self.points_count + 1; file.write(self.point(x0 + length, y0))
            p3_idx = self.points_count + 1; file.write(self.point(x0 + length, y0 + height))
            p4_idx = self.points_count + 1; file.write(self.point(x0, y0 + height))

            # Specify lines and capture their indices
            l1_idx = self.line_circs_count + 1; file.write(self.line(p1_idx, p2_idx))  # Bottom
            l2_idx = self.line_circs_count + 1; file.write(self.line(p2_idx, p3_idx))  # Right
            l3_idx = self.line_circs_count + 1; file.write(self.line(p3_idx, p4_idx))  # Top
            l4_idx = self.line_circs_count + 1; file.write(self.line(p4_idx, p1_idx))  # Left

            # Define surface
            file.write(self.curve_loop(l1_idx, l4_idx))
            file.write(self.plane_surface(self.curve_loop_count))
            """
            # This option doesn't work at the moment because of the quadliteral cell ordering, now using built
            in meshes instead.
            if cell_type == 'quad':
                # Calculate number of divisions
                nx = max(1, int(round(length / self.lc)))
                ny = max(1, int(round(height / self.lc)))

                # Add Transfinite and Recombine commands
                file.write(f"Transfinite Line {{{l1_idx}, {l3_idx}}} = {nx + 1};\n")
                file.write(f"Transfinite Line {{{l4_idx}, {l2_idx}}} = {ny + 1};\n")
                file.write(f"Transfinite Surface {{{self.plane_surface_count}}};\n")
                file.write(f"Recombine Surface {{{self.plane_surface_count}}};\n")
            """

            file.write("Mesh 2;\n")
            file.write(self.physical_surface(self.plane_surface_count))

        os.system(f"gmsh meshes/{file_name}.geo -2 -format msh2 -o meshes/{file_name}.msh")
        self.convert_to_xdmf(file_name, cell_type)

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
                                  mid_intersection: float, delta: float, h: float)->Tuple[Mesh, Mesh]:
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

        # Define the mesh resolution in the horizontal and vertical directions based on the element size
        nx = max(1, int(round(length / h)))
        ny_1 = max(1, int(round(height_1 / h)))
        ny_2 = max(1, int(round(height_2 / h)))

        # Create meshes directly using FEniCS
        mesh_rectangle_upper = RectangleMesh(p0_upper, p1_upper, nx, ny_1)
        mesh_rectangle_lower = RectangleMesh(p0_lower, p1_lower, nx, ny_2)
        return mesh_rectangle_upper, mesh_rectangle_lower

    def create_conforming_meshes(self, p0: Point, length: float, height: float,
                                 mid_intersection: float, delta: float, N_overlap: int=1)->Tuple[Mesh, Mesh]:
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
        y_bottom = p0[1]
        y_interface_lower = mid_intersection - delta / 2
        y_interface_upper = mid_intersection + delta / 2
        y_top = p0[1] + height

        # Define heights of the three distinct regions
        h_lower_region = y_interface_lower - y_bottom
        h_upper_region = y_top - y_interface_upper

        # Calculate number of cell divisions for each region
        ny_overlap = max(1, N_overlap)
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

        return  mesh_upper, mesh_lower


