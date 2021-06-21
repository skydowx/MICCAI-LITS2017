"""

Calculate nine common evaluation indicators of segmentation based on overlap and distance
"""

import math

import numpy as np
import scipy.spatial as spatial
import scipy.ndimage.morphology as morphology


class Metirc():
    
    def __init__(self, real_mask, pred_mask, voxel_spacing):
        """

        :param real_mask: gold standard
        :param pred_mask: forecast result
        :param voxel_spacing: Spacing of volume data
        """
        self.real_mask = real_mask
        self.pred_mask = pred_mask
        self.voxel_sapcing = voxel_spacing

        self.real_mask_surface_pts = self.get_surface(real_mask, voxel_spacing)
        self.pred_mask_surface_pts = self.get_surface(pred_mask, voxel_spacing)

        self.real2pred_nn = self.get_real2pred_nn()
        self.pred2real_nn = self.get_pred2real_nn()

    # 下面三个是提取边界和计算最小距离的实用函数
    def get_surface(self, mask, voxel_spacing):
        """

        :param mask: ndarray
        :param voxel_spacing: Spacing of volume data
        :return: Extract the real coordinates of the surface points of the array (in mm)
        """

        # The convolution kernel uses three-dimensional 18 neighborhoods

        kernel = morphology.generate_binary_structure(3, 2)
        surface = morphology.binary_erosion(mask, kernel) ^ mask

        surface_pts = surface.nonzero()

        surface_pts = np.array(list(zip(surface_pts[0], surface_pts[1], surface_pts[2])))

        # (0.7808688879013062, 0.7808688879013062, 2.5) (88, 410, 512)
        # The spacing and shape of the read data are not corresponding, so they need to be reversed.
        return surface_pts * np.array(self.voxel_sapcing[::-1]).reshape(1, 3)

    def get_pred2real_nn(self):
        """

        :return: The minimum distance from the predicted result surface voxel to the gold standard surface voxel
        """

        tree = spatial.cKDTree(self.real_mask_surface_pts)
        nn, _ = tree.query(self.pred_mask_surface_pts)

        return nn

    def get_real2pred_nn(self):
        """

        :return: The minimum distance from the gold standard 
        """
        tree = spatial.cKDTree(self.pred_mask_surface_pts)
        nn, _ = tree.query(self.real_mask_surface_pts)

        return nn

    # The following six indicators are based on overlap
    def get_dice_coefficient(self):
        """

        :return: dice coefficient numerator of dice coefficient denominator of dice coefficient (the latter two are used to calculate dice_global)
        """
        intersection = (self.real_mask * self.pred_mask).sum()
        union = self.real_mask.sum() + self.pred_mask.sum()

        return 2 * intersection / union, 2 * intersection, union

    def get_jaccard_index(self):
        """

        :return: Jaccard coefficient
        """
        intersection = (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return intersection / union

    def get_VOE(self):
        """

        :return: Volumetric Overlap Error
        """

        return 1 - self.get_jaccard_index()

    def get_RVD(self):
        """

        :return: Relative Volume Difference
        """

        return float(self.pred_mask.sum() - self.real_mask.sum()) / float(self.real_mask.sum())

    def get_FNR(self):
        """

        :return: False negative rate
        """
        fn = self.real_mask.sum() - (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return fn / union

    def get_FPR(self):
        """

        :return: False positive rate
        """
        fp = self.pred_mask.sum() - (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return fp / union

    # The following three indicators are based on distance
    def get_ASSD(self):
        """

        :return: Average Symmetric Surface Distance
        """
        return (self.pred2real_nn.sum() + self.real2pred_nn.sum()) / \
               (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0])

    def get_RMSD(self):
        """

        :return: Root Mean Square symmetric Surface Distance
        """
        return math.sqrt((np.power(self.pred2real_nn, 2).sum() + np.power(self.real2pred_nn, 2).sum()) /
                         (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0]))

    def get_MSD(self):
        """

        :return: Maximum Symmetric Surface Distance
        """
        return max(self.pred2real_nn.max(), self.real2pred_nn.max())
