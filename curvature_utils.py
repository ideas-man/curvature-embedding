import numpy as np
import trimesh as tm


def crv_measure(mesh, points, radius):
    """ Modified Discrete Mean Curvature Measure

    Returns the modified discrete mean curvature measure of a
    sphere centered at a point as detailed in ‘Restricted
    Delaunay triangulations and normal cycle’, Cohen-Steiner and Morvan.

    Only edges of 1-ring neighborhood are used.

    :mesh: Target mesh. Type: trimesh.Mesh.
    :points: Vertices of the target mesh. Type: trimesh.Points.
    :radius: Query ball radius. Type: float.
    :return mean_curv: Curvature values. Type: np.array.
    """
    points = np.asanyarray(points, dtype=np.float64)
    if not tm.util.is_shape(points, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    # axis aligned bounds
    bounds = np.column_stack((points - radius,
                              points + radius))

    # line segments that intersect axis aligned bounding box
    candidates = [list(mesh.face_adjacency_tree.intersection(b))
                  for b in bounds]

    mean_curv = np.empty(len(points))
    for i, (x, x_candidates) in enumerate(zip(points, candidates)):
        x_neighbors = mesh.vertices[mesh.vertex_neighbors[i]]
        ref_x_candidates = []
        for c in x_candidates:
            gen = mesh.vertices[mesh.face_adjacency_edges[c]]
            if np.any([gen[0] in x_neighbors, gen[1] in x_neighbors]):
                ref_x_candidates.append(c)
        endpoints = mesh.vertices[mesh.face_adjacency_edges[ref_x_candidates]]
        lengths = tm.curvature.line_ball_intersection(
            endpoints[:, 0],
            endpoints[:, 1],
            center=x,
            radius=radius)
        angles = mesh.face_adjacency_angles[ref_x_candidates]
        signs = np.where(mesh.face_adjacency_convex[ref_x_candidates], 1, -1)
        mean_curv[i] = (lengths * angles * signs).sum() / 2

    return mean_curv
