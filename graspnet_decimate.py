import argparse
import os

import trimesh as tm


def decimate(obj_id, data_path, dump_path, dec_faces):
    print("Obj: %03d" % obj_id)
    obj_path = os.path.join(data_path, 'models', '%03d' % obj_id, 'nontextured.ply')
    mesh = tm.load(obj_path)
    print("In: {}".format(mesh.vertices.shape[0]))
    mesh = mesh.simplify_quadratic_decimation(dec_faces)
    print("Out: {}".format(mesh.vertices.shape[0]))
    mesh.export(os.path.join(dump_path, '%03d.ply' % obj_id))
    print("Saved %03d.ply" % obj_id)


def manager(data_path, dump_path, num_verts):
    print("Decimate to: {}".format(num_verts))
    dec_faces = num_verts * 2
    obj_ids = range(88)
    for obj_id in obj_ids:
        decimate(obj_id, data_path, dump_path, dec_faces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to source GraspNet-1 Billion meshes.')
    parser.add_argument('--dump_path', type=str, help='Path to the dump directory.')
    parser.add_argument('--num_verts', type=int, default=1000, help='Target number of vertices.')
    FLAGS = parser.parse_args()

    manager(FLAGS.data_path, FLAGS.dump_path, FLAGS.num_verts)
