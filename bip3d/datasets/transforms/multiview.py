import copy
import random

import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose

from bip3d.registry import TRANSFORMS
from bip3d.structures.points import get_points_type
from ..utils import sample


@TRANSFORMS.register_module()
class MultiViewPipeline(BaseTransform):
    """HARD CODE"""
    def __init__(
        self,
        transforms,
        n_images,
        max_n_images=None,
        ordered=False,
        rotate_3rscan=False,
    ):
        super().__init__()
        self.transforms = Compose(transforms)
        self.n_images = n_images
        self.max_n_images = (
            max_n_images if max_n_images is not None else n_images
        )
        self.ordered = ordered
        self.rotate_3rscan = rotate_3rscan

    def transform(self, results: dict):
        """Transform function.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        imgs = []
        img_paths = []
        points = []
        intrinsics = []
        extrinsics = []
        depth_imgs = []
        trans_mat = []
        
        total_n = len(results["img_path"])
        sample_n = min(max(self.n_images, total_n), self.max_n_images)
        ids = sample(total_n, sample_n, self.ordered)
        for i in ids.tolist():
            _results = dict()
            _results["img_path"] = results["img_path"][i]
            if "depth_img_path" in results:
                _results["depth_img_path"] = results["depth_img_path"][i]
                if isinstance(results["depth_cam2img"], list):
                    _results["depth_cam2img"] = results["depth_cam2img"][i]
                    _results["cam2img"] = results["depth2img"]["intrinsic"][i]
                else:
                    _results["depth_cam2img"] = results["depth_cam2img"]
                    _results["cam2img"] = results["cam2img"]
                _results["depth_shift"] = results["depth_shift"]
            _results = self.transforms(_results)
            if "depth_shift" in _results:
                _results.pop("depth_shift")
            if "img" in _results:
                imgs.append(_results["img"])
                img_paths.append(_results["img_path"])
            if "depth_img" in _results:
                depth_imgs.append(_results["depth_img"])
            if "points" in _results:
                points.append(_results["points"])
            if _results.get("modify_cam2img"):
                intrinsics.append(_results["cam2img"])
            elif isinstance(results["depth2img"]["intrinsic"], list):
                intrinsics.append(results["depth2img"]["intrinsic"][i])
            else:
                intrinsics.append(results["depth2img"]["intrinsic"])
            extrinsics.append(results["depth2img"]["extrinsic"][i])
            if "trans_mat" in _results:
                trans_mat.append(_results["trans_mat"])
        for key in _results.keys():
            if key not in ["img", "points", "img_path"]:
                results[key] = _results[key]
        if len(imgs):
            if self.rotate_3rscan and "3rscan" in img_paths[0]:
                imgs = [np.transpose(x, (1, 0, 2)) for x in imgs]
                rot_mat = np.array(
                    [
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )
                rot_mat = [np.linalg.inv(x) @ rot_mat @ x for x in intrinsics]
                extrinsics = [x @ y for x, y in zip(rot_mat, extrinsics)]
                results["scale_factor"] = results["scale_factor"][::-1]
                results["ori_shape"] = results["ori_shape"][::-1]
            results["img"] = imgs
            results["img_path"] = img_paths
        if len(depth_imgs):
            if self.rotate_3rscan and "3rscan" in img_paths[0]:
                depth_imgs = [np.transpose(x, (1, 0)) for x in depth_imgs]
            results["depth_img"] = depth_imgs

        if len(points):
            results["points"] = points
        if (
            "ann_info" in results
            and "visible_instance_masks" in results["ann_info"]
        ):
            results["visible_instance_masks"] = [
                results["ann_info"]["visible_instance_masks"][i] for i in ids
            ]
            results["ann_info"]["visible_instance_masks"] = results[
                "visible_instance_masks"
            ]
        elif "visible_instance_masks" in results:
            results["visible_instance_masks"] = [
                results["visible_instance_masks"][i] for i in ids
            ]
        results["depth2img"]["intrinsic"] = intrinsics
        results["depth2img"]["extrinsic"] = extrinsics
        if len(trans_mat) != 0:
            results["depth2img"]["trans_mat"] = trans_mat
        return results
