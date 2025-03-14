from typing import List, Optional, Tuple, Union
import cv2
import copy

import numpy as np
from numpy import random
import mmcv
import torch
from PIL import Image
from mmcv.transforms import BaseTransform, Resize

from bip3d.registry import TRANSFORMS
from bip3d.structures.bbox_3d import points_cam2img, points_img2cam
from bip3d.structures.points import BasePoints, get_points_type


@TRANSFORMS.register_module()
class CategoryGroundingDataPrepare(BaseTransform):
    def __init__(
        self,
        classes,
        training,
        max_class=None,
        sep_token="[SEP]",
        filter_others=True,
        z_range=None,
        filter_invisible=False,
        instance_mask_key="visible_instance_masks",
    ):
        self.classes = list(classes)
        self.training = training
        self.max_class = max_class
        self.sep_token = sep_token
        self.filter_others = filter_others
        self.z_range = z_range
        self.filter_invisible = filter_invisible
        self.instance_mask_key = instance_mask_key

    def transform(self, results):
        visible_instance_masks = results.get(self.instance_mask_key)
        if isinstance(visible_instance_masks, (list, tuple)):
            visible_instance_masks = np.stack(visible_instance_masks).any(axis=0)
            
        ignore_mask = results.get("ignore_mask")
        gt_names = results["ann_info"]["gt_names"]
        if self.filter_others and "gt_labels_3d" in results:
            mask = results["gt_labels_3d"] >= 0
            results["gt_labels_3d"] = results["gt_labels_3d"][mask]
            results["gt_bboxes_3d"] = results["gt_bboxes_3d"][mask]
            visible_instance_masks = visible_instance_masks[mask]
            gt_names = [x for i, x in enumerate(gt_names) if mask[i]]
            if ignore_mask is not None:
                ignore_mask = ignore_mask[mask]

        if (
            self.z_range is not None
            and "gt_labels_3d" in results
            and self.training
        ):
            mask = torch.logical_and(
                results["gt_bboxes_3d"].tensor[..., 2] >= self.z_range[0],
                results["gt_bboxes_3d"].tensor[..., 2] <= self.z_range[1],
            ).numpy()
            results["gt_labels_3d"] = results["gt_labels_3d"][mask]
            results["gt_bboxes_3d"] = results["gt_bboxes_3d"][mask]
            visible_instance_masks = visible_instance_masks[mask]
            gt_names = [x for i, x in enumerate(gt_names) if mask[i]]
            if ignore_mask is not None:
                ignore_mask = ignore_mask[mask]

        if self.training or self.filter_invisible:
            results["gt_labels_3d"] = results["gt_labels_3d"][
                visible_instance_masks
            ]
            results["gt_bboxes_3d"] = results["gt_bboxes_3d"][
                visible_instance_masks
            ]
            gt_names = [
                x for i, x in enumerate(gt_names) if visible_instance_masks[i]
            ]
            if ignore_mask is not None:
                ignore_mask = ignore_mask[visible_instance_masks]
                results["ignore_mask"] = ignore_mask

        if self.training:
            if (
                self.max_class is not None
                and len(self.classes) > self.max_class
            ):
                classes = copy.deepcopy(gt_names)
                random.shuffle(self.classes)
                for c in self.classes:
                    if c in classes:
                        continue
                    classes.append(c)
                    if len(classes) >= self.max_class:
                        break
                random.shuffle(classes)
            else:
                classes = copy.deepcopy(self.classes)
        else:
            classes = copy.deepcopy(self.classes)
            gt_names = classes

        results["text"] = self.sep_token.join(classes)
        tokens_positive = []
        for name in gt_names:
            start = results["text"].find(
                self.sep_token + name + self.sep_token
            )
            if start == -1:
                if results["text"].startswith(name + self.sep_token):
                    start = 0
                else:
                    start = results["text"].find(self.sep_token + name) + len(
                        self.sep_token
                    )
            else:
                start += len(self.sep_token)
            end = start + len(name)
            tokens_positive.append([[start, end]])
        results["tokens_positive"] = tokens_positive
        return results


@TRANSFORMS.register_module()
class CamIntrisicStandardization(BaseTransform):
    def __init__(self, dst_intrinsic, dst_wh):
        if not isinstance(dst_intrinsic, np.ndarray):
            dst_intrinsic = np.array(dst_intrinsic)
        if dst_intrinsic.shape[0] == 3:
            tmp = np.eye(4)
            tmp[:3, :3] = dst_intrinsic
            dst_intrinsic = tmp
        self.dst_intrinsic = dst_intrinsic
        self.dst_wh = dst_wh
        u, v = np.arange(dst_wh[0]), np.arange(dst_wh[1])
        u = np.repeat(u[None], dst_wh[1], 0)
        v = np.repeat(v[:, None], dst_wh[0], 1)
        uv = np.stack([u, v, np.ones_like(u)], axis=-1)
        self.dst_pts = uv @ np.linalg.inv(self.dst_intrinsic[:3, :3]).T

    def transform(self, results):
        src_intrinsic = results["cam2img"][:3, :3]
        src_uv = self.dst_pts @ src_intrinsic.T
        src_uv = src_uv.astype(np.float32)
        if "depth_img" in results and results["img"].shape[:2] != results["depth_img"].shape[:2]:
            results["depth_img"] = cv2.resize(
                results["depth_img"], results["img"].shape[:2][::-1],
                interpolation=cv2.INTER_LINEAR,
            )
        for key in ["img", "depth_img"]:
            if key not in results:
                continue
            results[key] = cv2.remap(
                results[key],
                src_uv[..., 0],
                src_uv[..., 1],
                cv2.INTER_NEAREST,
            )
            # warp_mat = self.dst_intrinsic[:3, :3] @ np.linalg.inv(src_intrinsic)
            # cv2.warpAffine(results[key], warp_mat[:2, :3], self.dst_wh)
        results["cam2img"] = copy.deepcopy(self.dst_intrinsic)
        results["depth_cam2img"] = copy.deepcopy(self.dst_intrinsic)
        results["scale"] = self.dst_wh
        results['img_shape'] = results["img"].shape[:2]
        results['scale_factor'] = (1, 1)
        results['keep_ratio'] = False
        results["modify_cam2img"] = True
        return results


@TRANSFORMS.register_module()
class CustomResize(Resize):
    def _resize_img(self, results: dict, key="img"):
        """Resize images with ``results['scale']``."""

        if results.get(key, None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results[key] = img
            if key == "img":
                results['img_shape'] = img.shape[:2]
                results['scale_factor'] = (w_scale, h_scale)
                results['keep_ratio'] = self.keep_ratio

    def transform(self, results: dict):
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1],
                                           self.scale_factor)  # type: ignore
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_seg(results)
        self._resize_keypoints(results)
        if "depth_img" in results:
            self._resize_img(results, key="depth_img")
        return results


@TRANSFORMS.register_module()
class DepthProbLabelGenerator(BaseTransform):
    def __init__(
        self,
        max_depth=10,
        min_depth=0.25,
        num_depth=64,
        stride=[8, 16, 32, 64],
        origin_stride=1,
    ):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.num_depth = num_depth
        self.stride = [x//origin_stride for x in stride]
        self.origin_stride = origin_stride

    def transform(self, input_dict):
        depth = input_dict["inputs"]["depth_img"].cpu().numpy()
        if self.origin_stride != 1:
            H, W = depth.shape[-2:]
            depth = np.transpose(depth, (0, 2, 3, 1))
            depth = [mmcv.imresize(
                x, (W//self.origin_stride, H//self.origin_stride),
                interpolation="nearest",
            ) for x in depth]
            depth = np.stack(depth)[:,None]
        depth = np.clip(
            depth,
            a_min=self.min_depth,
            a_max=self.max_depth,
        )
        depth_anchor = np.linspace(
            self.min_depth, self.max_depth, self.num_depth)[:, None, None]
        distance = np.abs(depth - depth_anchor)
        mask = distance < (depth_anchor[1] - depth_anchor[0])
        depth_gt = np.where(mask, depth_anchor, 0)
        y = depth_gt.sum(axis=1, keepdims=True) - depth_gt
        depth_valid_mask = depth > 0
        depth_prob_gt = np.where(
            (depth_gt != 0) & depth_valid_mask,
            (depth - y) / (depth_gt - y),
            0,
        )
        views, _, H, W = depth.shape
        gt = []
        gt_map = []
        for s in self.stride:
            gt_tmp = np.reshape(
                depth_prob_gt, (views, self.num_depth, H//s, s, W//s, s))
            gt_tmp = gt_tmp.sum(axis=-1).sum(axis=3)
            mask_tmp = depth_valid_mask.reshape(views, 1, H//s, s, W//s, s)
            mask_tmp = mask_tmp.sum(axis=-1).sum(axis=3)
            gt_tmp /= np.clip(mask_tmp, a_min=1, a_max=None)
            # gt_map.append(np.transpose(gt_tmp, (0, 2, 3, 1)))
            gt_tmp = gt_tmp.reshape(views, self.num_depth, -1)
            gt_tmp = np.transpose(gt_tmp, (0, 2, 1))
            gt.append(gt_tmp)
        gt = np.concatenate(gt, axis=1)
        gt = np.clip(gt, a_min=0.0, a_max=1.0)
        input_dict["inputs"]["depth_prob_gt"] = torch.from_numpy(gt).to(
            input_dict["inputs"]["depth_img"])
        return input_dict
