import os
import meshio
from dolfin import *
import fenics

class GeometryParser:
    """
    This helper class generates the .geo-file to obtain meshes for basic geometries.
    """
    def __init__(self, lc):
        self.lc = lc  # characteristic length of the elements
        self.points_count = 0
        self.line_circs_count = 0
        self.curve_loop_count = 0
        self.physical_curve_count = 0
        self.plane_surface_count = 0
        self.physical_surface_count = 0

    def initialize_file(self, file_name):
        with open(f"meshes/{file_name}.geo", "w") as file:
            file.write("//parsed geo-file\n")

    def define_variables(self, file_name):
        with open(f"meshes/{file_name}.geo", "a") as file:
            file.write("\n")
            file.write(f"lc = {self.lc};\n")

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

    def rectangle_mesh(self, left_bottom_corner, length, height, file_name, cell_type="triangle"):
        self.points_count = 0
        self.line_circs_count = 0
        self.curve_loop_count = 0
        self.plane_surface_count = 0
        self.physical_surface_count = 0

        self.initialize_file(file_name)
        self.define_variables(file_name)
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

    def circle_mesh(self, center, radius, file_name):
        self.initialize_file(file_name)
        self.define_variables(file_name)
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