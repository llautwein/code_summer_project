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
                    print("Gamma_1 marked!")

        bbt1 = self.mesh1.bounding_box_tree()

        boundary_markers_2 = MeshFunction("size_t", self.mesh2, 1)
        boundary_markers_2.set_all(0)
        all_boundary.mark(boundary_markers_2, 1)

        for facet in facets(self.mesh2):
            if boundary_markers_2[facet] == 1:
                p = facet.midpoint()
                if bbt1.compute_first_entity_collision(p) < self.mesh1.num_cells():
                    boundary_markers_2[facet] = 2
                    print("Gamma_2 marked!")

        return boundary_markers_1, boundary_markers_2
