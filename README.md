# Curvature Embedding Experiments

This repo contains scripts and utils used for curvature embedding of the ACRONYM and GraspNet-1 Billion datasets.

Main scripts:
* `embed_acronym.py` - embedding script for the [ACRONYM](https://github.com/NVlabs/acronym) dataset.
* `embed_graspnet.py` - embedding script for the [GraspNet-1 Billion](https://graspnet.net) dataset.

Utils:
* `acronym_utils.py` - Util(s) for the ACRONYM dataset embedding. Credit: [Martin Sundermeyer](https://github.com/MartinSmeyer).
* `graspnet_utils.py` - Util(s) for the GraspNet-1 Billion embedding. Credit: [GraspNetAPI](https://github.com/graspnet/graspnetAPI).
* `curvature_utils.py` - defines the Modified Discrete Mean Curvature Measure (MDMCM) based on the DMCM as detailed in `Restricted
    Delaunay triangulations and normal cycle`, Cohen-Steiner and Morvan.
* `graspnet_decimate.py` - script for the decimation of the GraspNet-1 Billion source meshes.
