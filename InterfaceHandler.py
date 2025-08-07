from dolfin import *

class AllBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class InterfaceHandler:
    def __init__(self, mesh1: Mesh, mesh2: Mesh):
        self.mesh1 = mesh1
        self.mesh2 = mesh2


    def mark_interface_boundaries(self, dim):
        all_boundary = AllBoundary()

        bbt2 = self.mesh2.bounding_box_tree()

        boundary_markers_1 = MeshFunction("size_t", self.mesh1, dim-1)
        boundary_markers_1.set_all(0)
        all_boundary.mark(boundary_markers_1, 1)

        for facet in facets(self.mesh1):
            if boundary_markers_1[facet] == 1:
                p = facet.midpoint()
                if bbt2.compute_first_entity_collision(p) < self.mesh2.num_cells():
                    boundary_markers_1[facet.index()] = 2

        bbt1 = self.mesh1.bounding_box_tree()

        boundary_markers_2 = MeshFunction("size_t", self.mesh2, dim-1)
        boundary_markers_2.set_all(0)
        all_boundary.mark(boundary_markers_2, 1)

        for facet in facets(self.mesh2):
            if boundary_markers_2[facet] == 1:
                p = facet.midpoint()
                if bbt1.compute_first_entity_collision(p) < self.mesh1.num_cells():
                    boundary_markers_2[facet.index()] = 2

        return boundary_markers_1, boundary_markers_2


class OverlappingRectanglesInterfaceHandler:
    """
    Handles the marking of boundaries for two overlapping rectangular subdomains.

    This class assumes the subdomains are created as two independent meshes.
    It uses SubDomain classes to identify the physical and artificial interface
    boundaries based on their y-coordinates.
    """
    def __init__(self, mesh_upper: Mesh, mesh_lower: Mesh):
        self.mesh_upper = mesh_upper
        self.mesh_lower = mesh_lower

    def mark_interface_boundaries(self, y_coord_upper_interface, y_coord_lower_interface):
        """
        Marks boundaries for two overlapping rectangles.

        :param y_coord_upper_interface: The y-coordinate of the interface on the upper mesh (its bottom boundary).
        :param y_coord_lower_interface: The y-coordinate of the interface on the lower mesh (its top boundary).
        """

        class AllBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary

        class TopRectangleArtificialBoundary(SubDomain):
            def inside(self, x, on_boundary):
                # The interface is the bottom edge of the upper rectangle
                # near() compares floating point numbers with a tolerance
                return on_boundary and near(x[1], y_coord_upper_interface)

        class LowerRectangleArtificialBoundary(SubDomain):
            def inside(self, x, on_boundary):
                # The interface is the top edge of the lower rectangle
                return on_boundary and near(x[1], y_coord_lower_interface)

        all_boundary = AllBoundary()
        boundary_markers_upper = MeshFunction("size_t", self.mesh_upper, self.mesh_upper.topology().dim() - 1)
        boundary_markers_upper.set_all(0)
        all_boundary.mark(boundary_markers_upper, 1)

        # Instantiate and mark
        top_interface_marker = TopRectangleArtificialBoundary()
        top_interface_marker.mark(boundary_markers_upper, 2)

        # --- Instantiate and mark for lower mesh ---
        boundary_markers_lower = MeshFunction("size_t", self.mesh_lower, self.mesh_lower.topology().dim() - 1)
        boundary_markers_lower.set_all(0)
        all_boundary.mark(boundary_markers_lower, 1)

        lower_interface_marker = LowerRectangleArtificialBoundary()
        lower_interface_marker.mark(boundary_markers_lower, 2)

        return boundary_markers_upper, boundary_markers_lower

    def mark_interface_boundaries_3d(self, z_coord_upper_interface, z_coord_lower_interface):
        """
        Marks boundaries for two overlapping rectangles.

        :param y_coord_upper_interface: The y-coordinate of the interface on the upper mesh (its bottom boundary).
        :param y_coord_lower_interface: The y-coordinate of the interface on the lower mesh (its top boundary).
        """

        class AllBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary

        class TopCuboidArtificialBoundary(SubDomain):
            def inside(self, x, on_boundary):
                # The interface is the bottom face of the upper cuboid
                return on_boundary and near(x[2], z_coord_upper_interface)

        class LowerCuboidArtificialBoundary(SubDomain):
            def inside(self, x, on_boundary):
                # The interface is the top face of the lower cuboid
                return on_boundary and near(x[2], z_coord_lower_interface)

        all_boundary = AllBoundary()
        # Mark all boundaries as 1 (physical) first
        boundary_markers_upper = MeshFunction("size_t", self.mesh_upper, 2)
        boundary_markers_upper.set_all(0)
        all_boundary.mark(boundary_markers_upper, 1)

        # Overwrite the interface boundary with marker 2 (artificial)
        top_interface_marker = TopCuboidArtificialBoundary()
        top_interface_marker.mark(boundary_markers_upper, 2)

        # --- Repeat for the lower mesh ---
        boundary_markers_lower = MeshFunction("size_t", self.mesh_lower, 2)
        boundary_markers_lower.set_all(0)
        all_boundary.mark(boundary_markers_lower, 1)

        lower_interface_marker = LowerCuboidArtificialBoundary()
        lower_interface_marker.mark(boundary_markers_lower, 2)

        return boundary_markers_upper, boundary_markers_lower





