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

    def convert_to_xdmf(self, file_name):

        msh = meshio.read(f"meshes/{file_name}.msh")
        msh.write(f"meshes/{file_name}.xdmf")
        print("Converted mesh to XDMF format")

    def load_mesh(self, file_name):
        mesh = fenics.Mesh()
        with fenics.XDMFFile(f"meshes/{file_name}.xdmf") as xdmf_file:
            xdmf_file.read(mesh)
        return mesh

    def rectangle_mesh(self, left_bottom_corner, length, height, file_name):
        x0, y0 = left_bottom_corner
        points = [
            self.point(x0, y0),
            self.point(x0 + length, y0),
            self.point(x0 + length, y0 + height),
            self.point(x0, y0 + height)
        ]
        lines = [
            self.line(self.points_count - 3, self.points_count - 2),
            self.line(self.points_count - 2, self.points_count - 1),
            self.line(self.points_count - 1, self.points_count),
            self.line(self.points_count, self.points_count - 3)
        ]
        loop = self.curve_loop(
            1, self.line_circs_count)
        plane_surface = self.plane_surface(self.curve_loop_count)
        physical_surface = self.physical_surface(self.plane_surface_count)
        physical_boundary = self.physical_curve(
            [self.line_circs_count - 3, self.line_circs_count - 2, self.line_circs_count - 1, self.line_circs_count]
        )

        self.initialize_file(file_name)
        self.define_variables(file_name)
        with open(f"meshes/{file_name}.geo", "a") as file:
            file.write("\n// Rectangle Geometry\n")
            for p in points:
                file.write(p)
            for l in lines:
                file.write(l)
            file.write(loop)
            file.write(plane_surface)
            file.write("Mesh 2;\n")
            file.write(physical_surface)
            #file.write(physical_boundary)

        os.system(f"gmsh meshes/{file_name}.geo -2")
        self.convert_to_xdmf(file_name)

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