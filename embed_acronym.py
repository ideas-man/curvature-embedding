import argparse
import os
from itertools import repeat
from multiprocessing import Pool
from time import time

import numpy as np
import trimesh as tm

from acronym_utils import load_scales
from curvature_utils import crv_measure


def embed(scene_id, data_path, dump_path, obj_scales, crv_bounds, rad, m_type, subdivide):
    t_start = time()
    scene_path = os.path.join(data_path, 'scene_grasps_fork_01')

    scene_data = np.load(os.path.join(scene_path, '%06d.npz' % scene_id))
    obj_paths = scene_data['obj_paths']
    obj_transforms = scene_data['obj_transforms']

    scene_meshes = [tm.load(os.path.join(data_path, obj_path)) for obj_path in obj_paths]

    for mesh_id in range(len(scene_meshes)):
        obj_scale = obj_scales[obj_paths[mesh_id]]
        scene_meshes[mesh_id].apply_scale(obj_scale)
        mesh_mean = np.mean(scene_meshes[mesh_id].vertices, 0, keepdims=True)
        scene_meshes[mesh_id].vertices -= mesh_mean
        scene_meshes[mesh_id].apply_transform(obj_transforms[mesh_id])

    mesh = tm.util.concatenate(scene_meshes)
    if subdivide:
        (mesh.vertices, mesh.faces) = tm.remesh.subdivide(mesh.vertices, mesh.faces)

    crv = crv_measure(mesh, mesh.vertices, rad)

    crv[crv < crv_bounds[0]] = crv_bounds[0]
    crv[crv > crv_bounds[1]] = crv_bounds[1]
    channels = np.where(crv >= 0, True, False)
    crv[~channels] = 1 - (crv[~channels] - crv_bounds[0])/(0 - crv_bounds[0])
    crv[channels] = (crv[channels] - 0)/(crv_bounds[1] - 0)
    colors = [[crv[crv_id], 0, 0] if channels[crv_id] else [0, crv[crv_id], 0] for crv_id in range(crv.shape[0])]
    mesh.visual.vertex_colors = np.array(colors)

    tm.exchange.ply.export_ply(mesh)
    mesh.export(os.path.join(dump_path, '%06d.ply' % scene_id))
    t_end = time()
    print("SCENE: {}, TIME: {}".format(scene_id, t_end - t_start))


def manager(data_path, dump_path, crv_bound, rad, subdivide, num_workers):
    crv_bounds = [-crv_bound, crv_bound]
    scene_ids = range(len(os.listdir(os.path.join(data_path, 'scene_grasps_fork_01'))))
    obj_scales, _ = load_scales(data_path)
    with Pool(processes=num_workers) as pool:
        pool.starmap(embed, zip(scene_ids,
                                repeat(data_path),
                                repeat(dump_path),
                                repeat(obj_scales),
                                repeat(crv_bounds),
                                repeat(rad),
                                repeat(subdivide)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the ACRONYM dataset.')
    parser.add_argument('--dump_path', type=str, help='Path to the dump directory.')
    parser.add_argument('--crv_bound', type=float, default=0.1, help='Global curvature normalization bound.')
    parser.add_argument('--rad', type=float, default=0.01, help='Query ball radius.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers.')
    parser.add_argument('--subdivide', action='store_true', default=False, help='Scene mesh subdivision flag.')
    FLAGS = parser.parse_args()

    manager(FLAGS.data_path, FLAGS.dump_path, FLAGS.crv_bound, FLAGS.rad, FLAGS.subdivide, FLAGS.num_workers)
