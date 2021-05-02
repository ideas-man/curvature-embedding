
import xml.etree.ElementTree as ET
from transforms3d.euler import quat2euler, euler2mat
import numpy as np


class xmlReader():
    """
    Simple XML reader from GraspNet-1 Billion API.
    """

    def __init__(self, xmlfilename):
        self.xmlfilename = xmlfilename
        etree = ET.parse(self.xmlfilename)
        self.top = etree.getroot()

    def getposevectorlist(self):
        """ Parses pose vector of format [objectid, x, y, z, alpha, beta, gamma] """
        posevectorlist = []
        for i in range(len(self.top)):
            objectid = int(self.top[i][0].text)
            objectname = self.top[i][1].text
            objectpath = self.top[i][2].text
            translationtext = self.top[i][3].text.split()
            translation = []
            for text in translationtext:
                translation.append(float(text))
            quattext = self.top[i][4].text.split()
            quat = []
            for text in quattext:
                quat.append(float(text))
            alpha, beta, gamma = quat2euler(quat)
            x, y, z = translation
            alpha *= (180.0 / np.pi)
            beta *= (180.0 / np.pi)
            gamma *= (180.0 / np.pi)
            posevectorlist.append([objectid, x, y, z, alpha, beta, gamma])

        return posevectorlist


def parse_posevector(posevector):
    """ Parses the pose vector.

    :posevector: Pose vector [objectid, x, y, z, alpha, beta, gamma]. Type: list.
    :return obj_idx: Object ID. Type: int.
    :return mat: 4x4 pose matrix. Type: np.array.
    """
    mat = np.zeros([4, 4], dtype=np.float32)
    alpha, beta, gamma = posevector[4:7]
    alpha = alpha / 180.0 * np.pi
    beta = beta / 180.0 * np.pi
    gamma = gamma / 180.0 * np.pi
    mat[:3, :3] = euler2mat(alpha, beta, gamma)
    mat[:3, 3] = posevector[1:4]
    mat[3, 3] = 1
    obj_idx = int(posevector[0])

    return obj_idx, mat


def transform_points(points, trans):
    """ Transform a set of points using a 4x4 transformation matrix.

    :points: Set of points. Type: np.array.
    :trans: Transformation matrix. Type: np.array.
    :return points_trans: Transformed points. Type: np.array.
    """
    ones = np.ones([points.shape[0], 1], dtype=points.dtype)
    points_ = np.concatenate([points, ones], axis=-1)
    points_ = np.matmul(trans, points_.T).T
    points_trans = points_[:, :3]

    return points_trans
