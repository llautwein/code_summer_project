from dolfin import *

class AllBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class InterfaceHandler:
    def __init__(self, mesh1: Mesh, mesh2: Mesh):
        self.mesh1 = mesh1
        self.mesh2 = mesh2


    def mark_interface_boundaries(self):
        all_boundary = AllBoundary()

        bbt2 = self.mesh2.bounding_box_tree()

        boundary_markers_1 = MeshFunction("size_t", self.mesh1, 1)
        boundary_markers_1.set_all(0)
        all_boundary.mark(boundary_markers_1, 1)

        for facet in facets(self.mesh1):
            if boundary_markers_1[facet] == 1:
                p = facet.midpoint()
                if bbt2.compute_first_entity_collision(p) < self.mesh2.num_cells():
                    boundary_markers_1[facet] = 2

        bbt1 = self.mesh1.bounding_box_tree()

        boundary_markers_2 = MeshFunction("size_t", self.mesh2, 1)
        boundary_markers_2.set_all(0)
        all_boundary.mark(boundary_markers_2, 1)

        for facet in facets(self.mesh2):
            if boundary_markers_2[facet] == 1:
                p = facet.midpoint()
                if bbt1.compute_first_entity_collision(p) < self.mesh1.num_cells():
                    boundary_markers_2[facet] = 2

        return boundary_markers_1, boundary_markers_2

class OverlappingRectanglesInterfaceHandler:
    def __init__(self, mesh_upper: Mesh, mesh_lower: Mesh):
        self.mesh_upper = mesh_upper
        self.mesh_lower = mesh_lower

    def mark_interface_boundaries(self, floor_upper, height_lower):
        class AllBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary

        class TopRectangleArtificialBoundary(SubDomain):
            def inside(self, x, on_boundary):
                # Bottom edge, i.e. interface boundary of upper rectangle
                return on_boundary and near(x[1], floor_upper)

        boundary_markers_upper = MeshFunction("size_t", self.mesh_upper, 1)
        boundary_markers_upper.set_all(0)
        all_boundary = AllBoundary()
        trab = TopRectangleArtificialBoundary()
        all_boundary.mark(boundary_markers_upper, 1)
        trab.mark(boundary_markers_upper, 2)

        class LowerRectangleArtificialBoundary(SubDomain):
            def inside(self, x, on_boundary):
                # Bottom edge, i.e. interface boundary of lower rectangle
                return on_boundary and near(x[1], height_lower)

        boundary_markers_lower = MeshFunction("size_t", self.mesh_lower, 1)
        boundary_markers_lower.set_all(0)
        lrab = LowerRectangleArtificialBoundary()
        all_boundary.mark(boundary_markers_lower, 1)
        lrab.mark(boundary_markers_lower, 2)

        return boundary_markers_upper, boundary_markers_lower




