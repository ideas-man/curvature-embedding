import argparse
import os
from multiprocessing import Pool

import imageio
import numpy as np
import pyrender
import trimesh as tm

from curvature_utils import crv_measure
from graspnet_utils import xmlReader, parse_posevector, transform_points


def embed(scene_id, data_path, dump_path, model_path, crv_bound, rad, m_type):
    crv_bounds = [-crv_bound, crv_bound]
    K = {"fx": 631.54864502,
         "fy": 631.20751953,
         "cx": 638.43517329,
         "cy": 366.49904066,
         "znear": 0.04,
         "zfar": 20}
    height = 720
    width = 1280
    Rx = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    align_mat = np.load(os.path.join(data_path, 'scenes', 'scene_%04d' % scene_id, 'kinect', 'cam0_wrt_table.npy'))
    camera = pyrender.IntrinsicsCamera(K["fx"], K["fy"], K["cx"], K["cy"], K["znear"], K["zfar"])
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)

    if not os.path.exists(os.path.join(dump_path, 'scene_%04d' % scene_id)):
        os.makedirs(os.path.join(dump_path, 'scene_%04d' % scene_id))

    for ann_id in range(256):
        mesh_scene = pyrender.Scene()
        obj_paths = []
        obj_transforms = []

        camera_poses = np.load(os.path.join(data_path, 'scenes', 'scene_%04d' % scene_id, 'kinect', 'camera_poses.npy'))
        camera_pose = camera_poses[ann_id]
        camera_pose = np.matmul(align_mat, camera_pose)
        scene_reader = xmlReader(
            os.path.join(data_path, 'scenes', 'scene_%04d' % scene_id, 'kinect', 'annotations', '%04d.xml' % ann_id))
        posevectors = scene_reader.getposevectorlist()

        for posevector in posevectors:
            obj_id, obj_pose = parse_posevector(posevector)
            obj_pose = np.dot(camera_pose, obj_pose)
            obj_paths.append(os.path.join(model_path, '%03d.ply' % obj_id))
            obj_transforms.append(obj_pose)

        scene_meshes = [tm.load(os.path.join(data_path, obj_path)) for obj_path in obj_paths]

        for mesh_id in range(len(scene_meshes)):
            scene_meshes[mesh_id].vertices = transform_points(scene_meshes[mesh_id].vertices, obj_transforms[mesh_id])

        mesh = tm.util.concatenate(scene_meshes)

        crv = crv_measure(mesh, mesh.vertices, rad)

        crv[crv < crv_bounds[0]] = crv_bounds[0]
        crv[crv > crv_bounds[1]] = crv_bounds[1]
        channels = np.where(crv >= 0, True, False)
        crv[~channels] = 1 - (crv[~channels] - crv_bounds[0]) / (0 - crv_bounds[0])
        crv[channels] = (crv[channels] - 0) / (crv_bounds[1] - 0)
        colors = [[crv[crv_id], 0, 0] if channels[crv_id] else [0, crv[crv_id], 0] for crv_id in range(crv.shape[0])]
        mesh.visual.vertex_colors = np.array(colors)

        mesh_scene.add(pyrender.Mesh.from_trimesh(mesh=mesh))
        camera_pose = np.matmul(camera_pose, Rx)
        mesh_scene.add(camera, pose=camera_pose)

        color, _ = renderer.render(mesh_scene, flags=pyrender.constants.RenderFlags.FLAT)
        imageio.imwrite(os.path.join(dump_path, 'scene_%04d' % scene_id, '%04d.png' % ann_id), color)


def manager(data_path, dump_path, model_path, crv_bound, rad, num_workers):
    scene_ids = range(190)
    parpool = Pool(processes=num_workers)
    for scene_id in scene_ids:
        parpool.apply_async(embed, (scene_id, data_path, dump_path, model_path, crv_bound, rad))
    parpool.close()
    parpool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the GraspNet-1 Billion dataset.')
    parser.add_argument('--dump_path', type=str, help='Path to the dump directory.')
    parser.add_argument('--model_path', type=str, help='Path to source (decimated) models.')
    parser.add_argument('--crv_bound', type=float, default=0.1, help='Global curvature normalization bound.')
    parser.add_argument('--rad', type=float, default=0.01, help='Query ball radius.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers.')
    FLAGS = parser.parse_args()

    manager(FLAGS.data_path, FLAGS.dump_path, FLAGS.model_path, FLAGS.crv_bound, FLAGS.rad, FLAGS.num_workers)
