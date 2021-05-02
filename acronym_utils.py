import glob
import json
import os


def load_scales(data_path):
    """ Load categories and scales for the ACRONYM source meshes.

    :data_path: path to the ACRONYM root directory. Type: str.
    :return object_scales: Source mesh scales. Type: dict.
    :return object_cats: Object categories. Type: list.
    """
    all_mesh_scales = glob.glob(os.path.join(data_path, 'mesh_scales', '*'))
    object_scales = {}
    object_cats = set([])
    for mesh_scale in all_mesh_scales:
        with open(mesh_scale, 'r') as f:
            d = json.load(f)
            for k, v in d.items():
                object_scales[k] = v
                object_cats.add(k.split('/')[1])

    return object_scales, list(object_cats)
