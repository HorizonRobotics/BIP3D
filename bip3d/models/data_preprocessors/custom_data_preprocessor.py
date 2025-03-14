import math
from numbers import Number
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmdet.models import DetDataPreprocessor
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmengine.model import stack_batch
from mmengine.structures import InstanceData
from mmengine.utils import is_seq_of
from torch import Tensor
from torch.nn import functional as F

from bip3d.registry import MODELS
from bip3d.utils.typing_config import ConfigType, SampleList
from bip3d.structures.bbox_3d import get_proj_mat_by_coord_type

from .utils import multiview_img_stack_batch


@MODELS.register_module()
class CustomDet3DDataPreprocessor(DetDataPreprocessor):
    """Points / Image pre-processor for point clouds / vision-only / multi-
    modality 3D detection tasks.

    It provides the data pre-processing as follows

    - Collate and move image and point cloud data to the target device.

    - 1) For image data:

      - Pad images in inputs to the maximum size of current batch with defined
        ``pad_value``. The padding size can be divisible by a defined
        ``pad_size_divisor``.
      - Stack images in inputs to batch_imgs.
      - Convert images in inputs from bgr to rgb if the shape of input is
        (3, H, W).
      - Normalize images in inputs with defined std and mean.
      - Do batch augmentations during training.

    - 2) For point cloud data:

      - If no voxelization, directly return list of point cloud data.
      - If voxelization is applied, voxelize point cloud according to
        ``voxel_type`` and obtain ``voxels``.

    Args:
        voxel (bool): Whether to apply voxelization to point cloud.
            Defaults to False.
        voxel_type (str): Voxelization type. Two voxelization types are
            provided: 'hard' and 'dynamic', respectively for hard voxelization
            and dynamic voxelization. Defaults to 'hard'.
        voxel_layer (dict or :obj:`ConfigDict`, optional): Voxelization layer
            config. Defaults to None.
        batch_first (bool): Whether to put the batch dimension to the first
            dimension when getting voxel coordinates. Defaults to True.
        max_voxels (int, optional): Maximum number of voxels in each voxel
            grid. Defaults to None.
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be divisible by
            ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic segmentation
            maps. Defaults to 255.
        bgr_to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): Whether to convert image from RGB to BGR.
            Defaults to False.
        boxtype2tensor (bool): Whether to convert the ``BaseBoxes`` type of
            bboxes data to ``Tensor`` type. Defaults to True.
        non_blocking (bool): Whether to block current process when transferring
            data to device. Defaults to False.
        batch_augments (List[dict], optional): Batch-level augmentations.
            Defaults to None.
        batchwise_inputs (bool): Pack the input as a batch of samples
            with 1-N frames for the continuous 3D perception setting.
            Defaults to False.
    """

    def __init__(
        self,
        voxel: bool = False,
        voxel_type: str = "hard",
        voxel_layer: Optional[ConfigType] = None,
        batch_first: bool = True,
        max_voxels: Optional[int] = None,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        pad_mask: bool = False,
        mask_pad_value: int = 0,
        pad_seg: bool = False,
        seg_pad_value: int = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        boxtype2tensor: bool = True,
        non_blocking: bool = False,
        batch_augments: Optional[List[dict]] = None,
        batchwise_inputs: bool = False,
    ):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            pad_mask=pad_mask,
            mask_pad_value=mask_pad_value,
            pad_seg=pad_seg,
            seg_pad_value=seg_pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            boxtype2tensor=boxtype2tensor,
            non_blocking=non_blocking,
            batch_augments=batch_augments,
        )
        self.voxel = voxel
        self.voxel_type = voxel_type
        self.batch_first = batch_first
        self.max_voxels = max_voxels
        self.batchwise_inputs = batchwise_inputs
        if voxel:
            self.voxel_layer = VoxelizationByGridShape(**voxel_layer)

    def forward(self, data: Union[dict, List[dict]], training: bool = False):
        """Perform normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict or List[dict]): Data from dataloader. The dict contains
                the whole batch data, when it is a list[dict], the list
                indicates test time augmentation.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict or List[dict]: Data in the same format as the model input.
        """
        if isinstance(data, list):
            num_augs = len(data)
            aug_batch_data = []
            for aug_id in range(num_augs):
                single_aug_batch_data = self.simple_process(
                    data[aug_id], training
                )
                aug_batch_data.append(single_aug_batch_data)
            return aug_batch_data
        else:
            return self.simple_process(data, training)

    def simple_process(self, data: dict, training: bool = False):
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if "img" in data["inputs"]:
            batch_pad_shape = self._get_pad_shape(data)

        if self.batchwise_inputs:
            data_samples = data["data_samples"]
            batchwise_data_samples = []
            if "bboxes_3d" in data_samples[0].gt_instances_3d:
                assert isinstance(
                    data_samples[0].gt_instances_3d.labels_3d, list
                )
                bboxes_3d = data_samples[0].gt_instances_3d.bboxes_3d
                labels_3d = data_samples[0].gt_instances_3d.labels_3d
            if "gt_occupancy_masks" in data_samples[0]:
                gt_occupancy_masks = [
                    mask.clone() for mask in data_samples[0].gt_occupancy_masks
                ]
            if (
                "eval_ann_info" in data_samples[0]
                and data_samples[0].eval_ann_info is not None
            ):
                eval_ann_info = data_samples[0].eval_ann_info
            for idx in range(len(labels_3d)):
                data_sample = data_samples[0].clone()
                if "bboxes_3d" in data_sample.gt_instances_3d:
                    data_sample.gt_instances_3d = InstanceData()
                    data_sample.gt_instances_3d.bboxes_3d = bboxes_3d[idx]
                    data_sample.gt_instances_3d.labels_3d = labels_3d[idx]
                if "gt_occupancy_masks" in data_sample:
                    data_sample.gt_occupancy_masks = gt_occupancy_masks[idx]
                if "eval_ann_info" in data_sample:
                    if data_sample.eval_ann_info is not None:
                        data_sample.eval_ann_info = dict()
                        data_sample.eval_ann_info["gt_bboxes_3d"] = (
                            eval_ann_info["gt_bboxes_3d"][idx]
                        )
                        data_sample.eval_ann_info["gt_labels_3d"] = (
                            eval_ann_info["gt_labels_3d"][idx]
                        )
                batchwise_data_samples.append(data_sample)
            data["data_samples"] = batchwise_data_samples

        data = self.collate_data(data)
        inputs, data_samples = data["inputs"], data["data_samples"]
        batch_inputs = dict()
        batch_inputs.update(self.process_camera_params(data_samples))

        for key in ["depth_img", "depth_prob_gt"]:
            if key not in inputs:
                continue
            batch_inputs[key] = torch.stack(inputs[key])

        for key in ["text", "scan_id", "tokens_positive", "ignore_mask"]:
            if not hasattr(data_samples[0], key):
                continue
            batch_inputs[key] = [getattr(x, key) for x in data_samples]
        if hasattr(data_samples[0], "gt_instances_3d"):
            batch_inputs["gt_bboxes_3d"] = [
                x.gt_instances_3d.bboxes_3d for x in data_samples
            ]
            batch_inputs["gt_labels_3d"] = [
                x.gt_instances_3d.labels_3d for x in data_samples
            ]

        if "imgs" in inputs:
            imgs = inputs["imgs"]

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(
                    data_samples, batch_pad_shape
                ):
                    data_sample.set_metainfo(
                        {
                            "batch_input_shape": batch_input_shape,
                            "pad_shape": pad_shape,
                        }
                    )

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs["imgs"] = imgs

        return {"inputs": batch_inputs, "data_samples": data_samples}

    def process_camera_params(self, data_samples):
        projection_mat = []
        extrinsic_list = []
        intrinsic_list = []
        image_wh = []
        for data_sample in data_samples:
            proj_mat = get_proj_mat_by_coord_type(
                data_sample.metainfo, "DEPTH"
            )

            img_scale_factor = data_sample.metainfo.get("scale_factor", [1, 1])
            img_flip = data_sample.metainfo.get("flip", False)
            img_crop_offset = data_sample.metainfo.get(
                "img_crop_offset", [0, 0]
            )
            trans_mat = np.eye(4)
            trans_mat[0, 0] = img_scale_factor[0]
            trans_mat[1, 1] = img_scale_factor[1]
            trans_mat[0, 2] = -img_crop_offset[0]
            trans_mat[1, 2] = -img_crop_offset[1]
            if img_flip:
                assert False
            if "trans_mat" in proj_mat:
                trans_mat = np.stack(proj_mat["trans_mat"]) @ trans_mat

            if isinstance(proj_mat, dict):
                extrinsic = np.stack(proj_mat["extrinsic"])
                intrinsic = np.stack(proj_mat["intrinsic"])
                proj_mat = intrinsic @ extrinsic
                extrinsic_list.append(extrinsic)
                intrinsic_list.append(trans_mat @ intrinsic)
            else:
                extrinsic_list.append(
                    np.tile(np.eye(4)[None], (proj_mat.shape[0], 1, 1))
                )
                intrinsic_list.append(
                    np.tile(np.eye(4)[None], (proj_mat.shape[0], 1, 1))
                )
            proj_mat = trans_mat @ proj_mat
            projection_mat.append(proj_mat)
            image_wh.append(data_sample.metainfo["img_shape"][:2])

        to_tensor = lambda x: torch.from_numpy(x).cuda().to(torch.float32)
        projection_mat = to_tensor(np.stack(projection_mat))
        image_wh = to_tensor(np.array(image_wh))
        image_wh = image_wh[:, None].tile(1, projection_mat.shape[1], 1)
        extrinsic = to_tensor(np.stack(extrinsic_list))
        intrinsic = to_tensor(np.stack(intrinsic_list))
        return {
            "projection_mat": projection_mat,
            "image_wh": image_wh,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        }

    def preprocess_img(self, _batch_img: Tensor):
        # channel transform
        if self._channel_conversion:
            _batch_img = _batch_img[[2, 1, 0], ...]
        # Convert to float after channel conversion to ensure
        # efficiency
        _batch_img = _batch_img.float()
        # Normalization.
        if self._enable_normalize:
            if self.mean.shape[0] == 3:
                assert _batch_img.dim() == 3 and _batch_img.shape[0] == 3, (
                    "If the mean has 3 values, the input tensor "
                    "should in shape of (3, H, W), but got the "
                    f"tensor with shape {_batch_img.shape}"
                )
            _batch_img = (_batch_img - self.mean) / self.std
        return _batch_img

    def collate_data(self, data: dict):
        """Copy data to the target device and perform normalization, padding
        and bgr2rgb conversion and stack based on ``BaseDataPreprocessor``.

        Collates the data sampled from dataloader into a list of dict and list
        of labels, and then copies tensor to the target device.

        Args:
            data (dict): Data sampled from dataloader.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore

        if "img" in data["inputs"]:
            _batch_imgs = data["inputs"]["img"]
            # Process data with `pseudo_collate`.
            if is_seq_of(_batch_imgs, torch.Tensor):
                batch_imgs = []
                img_dim = _batch_imgs[0].dim()
                for _batch_img in _batch_imgs:
                    if img_dim == 3:  # standard img
                        _batch_img = self.preprocess_img(_batch_img)
                    elif img_dim == 4:
                        _batch_img = [
                            self.preprocess_img(_img) for _img in _batch_img
                        ]

                        _batch_img = torch.stack(_batch_img, dim=0)

                    batch_imgs.append(_batch_img)

                # Pad and stack Tensor.
                if img_dim == 3:
                    batch_imgs = stack_batch(
                        batch_imgs, self.pad_size_divisor, self.pad_value
                    )
                elif img_dim == 4:
                    batch_imgs = multiview_img_stack_batch(
                        batch_imgs, self.pad_size_divisor, self.pad_value
                    )

            # Process data with `default_collate`.
            elif isinstance(_batch_imgs, torch.Tensor):
                assert _batch_imgs.dim() == 4, (
                    "The input of `ImgDataPreprocessor` should be a NCHW "
                    "tensor or a list of tensor, but got a tensor with "
                    f"shape: {_batch_imgs.shape}"
                )
                if self._channel_conversion:
                    _batch_imgs = _batch_imgs[:, [2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_imgs = _batch_imgs.float()
                if self._enable_normalize:
                    _batch_imgs = (_batch_imgs - self.mean) / self.std
                h, w = _batch_imgs.shape[2:]
                target_h = (
                    math.ceil(h / self.pad_size_divisor)
                    * self.pad_size_divisor
                )
                target_w = (
                    math.ceil(w / self.pad_size_divisor)
                    * self.pad_size_divisor
                )
                pad_h = target_h - h
                pad_w = target_w - w
                batch_imgs = F.pad(
                    _batch_imgs,
                    (0, pad_w, 0, pad_h),
                    "constant",
                    self.pad_value,
                )
            else:
                raise TypeError(
                    "Output of `cast_data` should be a list of dict "
                    "or a tuple with inputs and data_samples, but got "
                    f"{type(data)}: {data}"
                )

            data["inputs"]["imgs"] = batch_imgs

        data.setdefault("data_samples", None)

        return data

    def _get_pad_shape(self, data: dict):
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        # rewrite `_get_pad_shape` for obtaining image inputs.
        _batch_inputs = data["inputs"]["img"]
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                if ori_input.dim() == 4:
                    # mean multiview input, select one of the
                    # image to calculate the pad shape
                    ori_input = ori_input[0]
                pad_h = (
                    int(np.ceil(ori_input.shape[1] / self.pad_size_divisor))
                    * self.pad_size_divisor
                )
                pad_w = (
                    int(np.ceil(ori_input.shape[2] / self.pad_size_divisor))
                    * self.pad_size_divisor
                )
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                "The input of `ImgDataPreprocessor` should be a NCHW tensor "
                "or a list of tensor, but got a tensor with shape: "
                f"{_batch_inputs.shape}"
            )
            pad_h = (
                int(np.ceil(_batch_inputs.shape[1] / self.pad_size_divisor))
                * self.pad_size_divisor
            )
            pad_w = (
                int(np.ceil(_batch_inputs.shape[2] / self.pad_size_divisor))
                * self.pad_size_divisor
            )
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError(
                "Output of `cast_data` should be a list of dict "
                "or a tuple with inputs and data_samples, but got "
                f"{type(data)}: {data}"
            )
        return batch_pad_shape
