import math
import pickle
import copy
import tqdm
import os
import warnings
from typing import Callable, List, Optional, Union

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.fileio import load
from mmengine.logging import print_log

from bip3d.registry import DATASETS
from bip3d.structures import get_box_type
from .utils import sample

class_names = (
    'adhesive tape', 'air conditioner', 'alarm', 'album', 'arch', 'backpack',
    'bag', 'balcony', 'ball', 'banister', 'bar', 'barricade', 'baseboard',
    'basin', 'basket', 'bathtub', 'beam', 'beanbag', 'bed', 'bench', 'bicycle',
    'bidet', 'bin', 'blackboard', 'blanket', 'blinds', 'board', 'body loofah',
    'book', 'boots', 'bottle', 'bowl', 'box', 'bread', 'broom', 'brush',
    'bucket', 'cabinet', 'calendar', 'camera', 'can', 'candle', 'candlestick',
    'cap', 'car', 'carpet', 'cart', 'case', 'chair', 'chandelier', 'cleanser',
    'clock', 'clothes', 'clothes dryer', 'coat hanger', 'coffee maker', 'coil',
    'column', 'commode', 'computer', 'conducting wire', 'container', 'control',
    'copier', 'cosmetics', 'couch', 'counter', 'countertop', 'crate', 'crib',
    'cube', 'cup', 'curtain', 'cushion', 'decoration', 'desk', 'detergent',
    'device', 'dish rack', 'dishwasher', 'dispenser', 'divider', 'door',
    'door knob', 'doorframe', 'doorway', 'drawer', 'dress', 'dresser', 'drum',
    'duct', 'dumbbell', 'dustpan', 'dvd', 'eraser', 'excercise equipment',
    'fan', 'faucet', 'fence', 'file', 'fire extinguisher', 'fireplace',
    'flowerpot', 'flush', 'folder', 'food', 'footstool', 'frame', 'fruit',
    'furniture', 'garage door', 'garbage', 'glass', 'globe', 'glove',
    'grab bar', 'grass', 'guitar', 'hair dryer', 'hamper', 'handle', 'hanger',
    'hat', 'headboard', 'headphones', 'heater', 'helmets', 'holder', 'hook',
    'humidifier', 'ironware', 'jacket', 'jalousie', 'jar', 'kettle',
    'keyboard', 'kitchen island', 'kitchenware', 'knife', 'label', 'ladder',
    'lamp', 'laptop', 'ledge', 'letter', 'light', 'luggage', 'machine',
    'magazine', 'mailbox', 'map', 'mask', 'mat', 'mattress', 'menu',
    'microwave', 'mirror', 'molding', 'monitor', 'mop', 'mouse', 'napkins',
    'notebook', 'ottoman', 'oven', 'pack', 'package', 'pad', 'pan', 'panel',
    'paper', 'paper cutter', 'partition', 'pedestal', 'pen', 'person', 'piano',
    'picture', 'pillar', 'pillow', 'pipe', 'pitcher', 'plant', 'plate',
    'player', 'plug', 'plunger', 'pool', 'pool table', 'poster', 'pot',
    'price tag', 'printer', 'projector', 'purse', 'rack', 'radiator', 'radio',
    'rail', 'range hood', 'refrigerator', 'remote control', 'ridge', 'rod',
    'roll', 'roof', 'rope', 'sack', 'salt', 'scale', 'scissors', 'screen',
    'seasoning', 'shampoo', 'sheet', 'shelf', 'shirt', 'shoe', 'shovel',
    'shower', 'sign', 'sink', 'soap', 'soap dish', 'soap dispenser', 'socket',
    'speaker', 'sponge', 'spoon', 'stairs', 'stall', 'stand', 'stapler',
    'statue', 'steps', 'stick', 'stool', 'stopcock', 'stove', 'structure',
    'sunglasses', 'support', 'switch', 'table', 'tablet', 'teapot',
    'telephone', 'thermostat', 'tissue', 'tissue box', 'toaster', 'toilet',
    'toilet paper', 'toiletry', 'tool', 'toothbrush', 'toothpaste', 'towel',
    'toy', 'tray', 'treadmill', 'trophy', 'tube', 'tv', 'umbrella', 'urn',
    'utensil', 'vacuum cleaner', 'vanity', 'vase', 'vent', 'ventilation',
    'wardrobe', 'washbasin', 'washing machine', 'water cooler', 'water heater',
    'window', 'window frame', 'windowsill', 'wine', 'wire', 'wood', 'wrap')
head_labels = [
    48, 177, 82, 179, 37, 243, 28, 277, 32, 84, 215, 145, 182, 170, 22, 72, 30,
    141, 65, 257, 221, 225, 52, 75, 231, 158, 236, 156, 47, 74, 6, 18, 71, 242,
    217, 251, 66, 263, 5, 45, 14, 73, 278, 198, 24, 23, 196, 252, 19, 135, 26,
    229, 183, 200, 107, 272, 246, 269, 125, 59, 279, 15, 163, 258, 57, 195, 51,
    88, 97, 58, 102, 36, 137, 31, 80, 160, 155, 61, 238, 96, 190, 25, 219, 152,
    142, 201, 274, 249, 178, 192
]
common_labels = [
    189, 164, 101, 205, 273, 233, 131, 180, 86, 220, 67, 268, 224, 270, 53,
    203, 237, 226, 10, 133, 248, 41, 55, 16, 199, 134, 99, 185, 2, 20, 234,
    194, 253, 35, 174, 8, 223, 13, 91, 262, 230, 121, 49, 63, 119, 162, 79,
    168, 245, 267, 122, 104, 100, 1, 176, 280, 140, 209, 259, 143, 165, 147,
    117, 85, 105, 95, 109, 207, 68, 175, 106, 60, 4, 46, 171, 204, 111, 211,
    108, 120, 157, 222, 17, 264, 151, 98, 38, 261, 123, 78, 118, 127, 240, 124
]
tail_labels = [
    76, 149, 173, 250, 275, 255, 34, 77, 266, 283, 112, 115, 186, 136, 256, 40,
    254, 172, 9, 212, 213, 181, 154, 94, 191, 193, 3, 130, 146, 70, 128, 167,
    126, 81, 7, 11, 148, 228, 239, 247, 21, 42, 89, 153, 161, 244, 110, 0, 29,
    114, 132, 159, 218, 232, 260, 56, 92, 116, 282, 33, 113, 138, 12, 188, 44,
    150, 197, 271, 169, 206, 90, 235, 103, 281, 184, 208, 216, 202, 214, 241,
    129, 210, 276, 64, 27, 87, 139, 227, 187, 62, 43, 50, 69, 93, 144, 166,
    265, 54, 83, 39
]


@DATASETS.register_module()
class EmbodiedScanDetGroundingDataset(BaseDataset):
    def __init__(
        self,
        data_root: str,
        ann_file: str,
        vg_file=None,
        metainfo: Optional[dict] = None,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        load_eval_anns: bool = True,
        filter_empty_gt: bool = True,
        remove_dontcare: bool = False,
        box_type_3d: str = "Euler-Depth",
        dataset_length=None,
        mode="detection",
        max_n_images=50,
        n_images_per_sample=1,
        drop_last_per_scene=False,
        part=None,
        temporal=False,
        num_text=1,
        tokens_positive_rebuild=True,
        sep_token="[SEP]",
        **kwargs,
    ):
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.filter_empty_gt = filter_empty_gt
        self.remove_dontcare = remove_dontcare
        self.load_eval_anns = load_eval_anns
        self.dataset_length = dataset_length
        self.part = part
        self.mode = mode
        assert self.mode in ["detection", "continuous", "grounding"]
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            pipeline=pipeline,
            test_mode=test_mode,
            serialize_data=self.mode == "detection",
            **kwargs,
        )
        if self.mode == "continuous":
            self.max_n_images = max_n_images
            self.n_images_per_sample = n_images_per_sample
            self.drop_last_per_scene = drop_last_per_scene
            self.convert_to_continuous()
        elif self.mode == "grounding":
            self.vg_file = vg_file
            self.num_text = num_text
            self.tokens_positive_rebuild = tokens_positive_rebuild
            self.sep_token = sep_token
            self.load_language_data()
            self.data_bytes, self.data_address = self._serialize_data()
            self.serialize_data = True
        print_log(f"dataset length : {self.__len__()}")

    def process_metainfo(self):
        assert "categories" in self._metainfo

        if "classes" not in self._metainfo:
            self._metainfo.setdefault(
                "classes", list(self._metainfo["categories"].keys())
            )

        self.label_mapping = np.full(
            max(list(self._metainfo["categories"].values())) + 1, -1, dtype=int
        )
        for key, value in self._metainfo["categories"].items():
            if key in self._metainfo["classes"]:
                self.label_mapping[value] = self._metainfo["classes"].index(
                    key
                )

    def parse_data_info(self, info: dict):
        info["box_type_3d"] = self.box_type_3d
        info["axis_align_matrix"] = self._get_axis_align_matrix(info)
        info["scan_id"] = info["sample_idx"]
        ann_dataset = info["sample_idx"].split("/")[0]
        if ann_dataset == "matterport3d":
            info["depth_shift"] = 4000.0
        else:
            info["depth_shift"] = 1000.0
        # Because multi-view settings are different from original designs
        # we temporarily follow the ori design in ImVoxelNet
        info["img_path"] = []
        info["depth_img_path"] = []
        if "cam2img" in info:
            cam2img = info["cam2img"].astype(np.float32)
        else:
            cam2img = []

        extrinsics = []
        for i in range(len(info["images"])):
            img_path = os.path.join(
                self.data_prefix.get("img_path", ""),
                info["images"][i]["img_path"],
            )
            depth_img_path = os.path.join(
                self.data_prefix.get("img_path", ""),
                info["images"][i]["depth_path"],
            )

            info["img_path"].append(img_path)
            info["depth_img_path"].append(depth_img_path)
            align_global2cam = np.linalg.inv(
                info["axis_align_matrix"] @ info["images"][i]["cam2global"]
            )
            extrinsics.append(align_global2cam.astype(np.float32))
            if "cam2img" not in info:
                cam2img.append(info["images"][i]["cam2img"].astype(np.float32))

        info["depth2img"] = dict(
            extrinsic=extrinsics,
            intrinsic=cam2img,
            origin=np.array([0.0, 0.0, 0.5]).astype(np.float32),
        )

        if "depth_cam2img" not in info:
            info["depth_cam2img"] = cam2img

        if not self.test_mode:
            info["ann_info"] = self.parse_ann_info(info)

        if self.test_mode and self.load_eval_anns:
            info["ann_info"] = self.parse_ann_info(info)
            info["eval_ann_info"] = self._remove_dontcare(info["ann_info"])

        return info

    def parse_ann_info(self, info: dict):
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`.
        """

        ann_info = None
        if "instances" in info and len(info["instances"]) > 0:
            ann_info = dict(
                gt_bboxes_3d=np.zeros(
                    (len(info["instances"]), 9), dtype=np.float32
                ),
                gt_labels_3d=np.zeros(
                    (len(info["instances"]),), dtype=np.int64
                ),
                gt_names=[],
                bbox_id=np.zeros((len(info["instances"]),), dtype=np.int64) - 1,
            )
            for idx, instance in enumerate(info["instances"]):
                ann_info["gt_bboxes_3d"][idx] = instance["bbox_3d"]
                ann_info["gt_labels_3d"][idx] = self.label_mapping[
                    instance["bbox_label_3d"]
                ]
                ann_info["gt_names"].append(
                    self._metainfo["classes"][ann_info["gt_labels_3d"][idx]]
                    if ann_info["gt_labels_3d"][idx] >= 0
                    else "others"
                )
                ann_info["bbox_id"][idx] = instance["bbox_id"]

        # pack ann_info for return
        if ann_info is None:
            ann_info = dict()
            ann_info["gt_bboxes_3d"] = np.zeros((0, 9), dtype=np.float32)
            ann_info["gt_labels_3d"] = np.zeros((0,), dtype=np.int64)
            ann_info["bbox_id"] = np.zeros((0,), dtype=np.int64) - 1
            ann_info["gt_names"] = []

        # post-processing/filtering ann_info if not empty gt
        if "visible_instance_ids" in info["images"][0]:
            ids = []
            for i in range(len(info["images"])):
                ids.append(info["images"][i]["visible_instance_ids"])
            mask_length = ann_info["gt_labels_3d"].shape[0]
            ann_info["visible_instance_masks"] = self._ids2masks(
                ids, mask_length
            )

        if self.remove_dontcare:
            ann_info = self._remove_dontcare(ann_info)

        ann_dataset = info["sample_idx"].split("/")[0]
        ann_info["gt_bboxes_3d"] = self.box_type_3d(
            ann_info["gt_bboxes_3d"],
            box_dim=ann_info["gt_bboxes_3d"].shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0.5),
        )
        return ann_info

    @staticmethod
    def _get_axis_align_matrix(info: dict):
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): Info of a single sample data.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if "axis_align_matrix" in info:
            return np.array(info["axis_align_matrix"])
        else:
            warnings.warn(
                "axis_align_matrix is not found in ScanNet data info, please "
                "use new pre-process scripts to re-generate ScanNet data"
            )
            return np.eye(4).astype(np.float32)

    def _ids2masks(self, ids, mask_length):
        """Change visible_instance_ids to visible_instance_masks."""
        masks = []
        for idx in range(len(ids)):
            mask = np.zeros((mask_length,), dtype=bool)
            mask[ids[idx]] = 1
            masks.append(mask)
        return masks

    def _remove_dontcare(self, ann_info: dict):
        """Remove annotations that do not need to be cared.

        -1 indicates dontcare in MMDet3d.

        Args:
            ann_info (dict): Dict of annotation infos. The
                instance with label `-1` will be removed.

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        filter_mask = ann_info["gt_labels_3d"] > -1
        for key in ann_info.keys():
            if key == "instances":
                img_filtered_annotations[key] = ann_info[key]
            elif key == "visible_instance_masks":
                img_filtered_annotations[key] = []
                for idx in range(len(ann_info[key])):
                    img_filtered_annotations[key].append(
                        ann_info[key][idx][filter_mask]
                    )
            elif key == "gt_names":
                img_filtered_annotations[key] = [
                    x for i, x in enumerate(ann_info[key]) if filter_mask[i]
                ]
            else:
                img_filtered_annotations[key] = ann_info[key][filter_mask]
        return img_filtered_annotations

    def load_data_list(self):
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(
                f"The annotations loaded from annotation file "
                f"should be a dict, but got {type(annotations)}!"
            )
        if "data_list" not in annotations or "metainfo" not in annotations:
            raise ValueError(
                "Annotation must have data_list and metainfo " "keys"
            )
        metainfo = annotations["metainfo"]
        raw_data_list = annotations["data_list"]

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        self.process_metainfo()

        # load and parse data_infos.
        data_list = []
        for raw_data_info in tqdm.tqdm(
            raw_data_list,
            mininterval=10,
            desc=f"Loading {'Test' if self.test_mode else 'Train'} dataset",
        ):
            if self.part is not None:
                valid = False
                for x in self.part:
                    if x in raw_data_info["sample_idx"]:
                        valid = True
                        break
                if not valid:
                    continue

            data_info = self.parse_data_info(raw_data_info)
            if data_info is None:
                continue
            assert isinstance(data_info, dict)
            data_list.append(data_info)

            if (
                self.dataset_length is not None
                and len(data_list) >= self.dataset_length
            ):
                break
        return data_list

    @staticmethod
    def _get_axis_align_matrix(info: dict):
        if 'axis_align_matrix' in info:
            return np.array(info['axis_align_matrix'])
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    @staticmethod
    def _is_view_dep(text):
        """Check whether to augment based on sr3d utterance."""
        rels = [
            'front', 'behind', 'back', 'left', 'right', 'facing', 'leftmost',
            'rightmost', 'looking', 'across'
        ]
        words = set(text.split())
        return any(rel in words for rel in rels)

    def convert_info_to_scan(self):
        self.scans = dict()
        for data in self.data_list:
            scan_id = data['scan_id']
            self.scans[scan_id] = data
        self.scan_ids = list(self.scans.keys())

    def load_language_data(self):
        self.convert_info_to_scan()
        if isinstance(self.vg_file, str):
            language_annotations = load(os.path.join(self.data_root, self.vg_file))
        else:
            language_annotations = []
            for x in self.vg_file:
                language_annotations.extend(load(os.path.join(self.data_root, x)))
        if self.dataset_length is not None:
            interval = len(language_annotations) / self.dataset_length
            output = []
            for i in range(self.dataset_length):
                output.append(ids[int(interval*i)])
            language_annotations = output
        self.data_list = language_annotations
        self.scan_id_to_data_idx = {}
        for scan_id in self.scan_ids:
            self.scan_id_to_data_idx[scan_id] = []
        for i, d in enumerate(self.data_list):
            self.scan_id_to_data_idx[d["scan_id"]].append(i)

    def convert_to_continuous(self):
        self.convert_info_to_scan()
        data_list = []
        self.flag = []
        self.image_id_dict = {}
        for flag, scan_id in enumerate(self.scan_ids):
            total_n = len(self.scans[scan_id]["images"])
            sample_n = min(self.max_n_images, total_n)
            ids = sample(total_n, sample_n, True).tolist()
            self.image_id_dict[scan_id] = ids
            if self.n_images_per_sample > 1:
                if self.drop_last_per_scene:
                    sample_n = math.floor(sample_n / self.n_images_per_sample)
                else:
                    sample_n = math.ceil(sample_n / self.n_images_per_sample)

                ids = [
                    ids[i*self.n_images_per_sample : (i+1)*self.n_images_per_sample]
                    for i in range(sample_n)
                ]
            data_list.extend(
                [dict(scan_id=scan_id, image_id=i) for i in ids]
            )
            self.flag.extend([flag] * len(ids))
        self.data_list = data_list
        self.flag = np.array(self.flag)

    def get_data_info_continuous(self, data_info):
        scan_id = data_info["scan_id"]
        data = copy.deepcopy(self.scans[scan_id])
        img_idx = data_info["image_id"]
        if isinstance(img_idx, int):
            img_idx = [img_idx]
        for k in ["images", "img_path", "depth_img_path"]:
            data[k] = index(data[k], img_idx)

        seen_image_id = self.image_id_dict[scan_id]
        seen_image_id = seen_image_id[:seen_image_id.index(img_idx[0])]

        if len(seen_image_id) != 0:
            last_visible_mask = index(
                data["ann_info"]["visible_instance_masks"], seen_image_id
            )
            last_visible_mask = np.stack(last_visible_mask).any(axis=0)
        else:
            last_visible_mask = data["ann_info"]["visible_instance_masks"][0] * False

        visible_instance_masks = index(
            data["ann_info"]["visible_instance_masks"], img_idx
        )

        current_visible_mask = np.stack(visible_instance_masks).any(axis=0)
        ignore_mask = np.logical_and(
            last_visible_mask, ~current_visible_mask
        )
        all_visible_mask = np.logical_or(
            last_visible_mask, current_visible_mask,
        )
        data["visible_instance_masks"] = visible_instance_masks
        data["all_visible_mask"] = all_visible_mask
        data["ignore_mask"] = ignore_mask

        data["depth2img"]["extrinsic"] = index(
            data["depth2img"]["extrinsic"], img_idx
        )
        if isinstance(data["depth2img"]["intrinsic"], list):
            data["depth2img"]["intrinsic"] = index(
                data["depth2img"]["intrinsic"], img_idx
            )
        return data

    def merge_grounding_data(self, data_infos):
        output = dict(
            text="",
        )
        for key in ["target_id", "distractor_ids", "target", "anchors", "anchor_ids", "tokens_positive"]:
            if key in data_infos[0]:
                output[key] = []
        for data_info in data_infos:
            if "target_id" in data_info and data_info["target_id"] in output["target_id"]:
                continue

            if self.tokens_positive_rebuild and "target" in data_info:
                start_idx = data_info["text"].find(data_info["target"])
                end_idx = start_idx + len(data_info["target"])
                tokens_positive = [[start_idx, end_idx]]
            elif "tokens_positive" in data_info:
                tokens_positive = data_info["tokens_positive"]
            else:
                tokens_positive = None

            if len(output["text"]) == 0:
                output["text"] = data_info["text"]
            else:
                if tokens_positive is not None:
                    tokens_positive = np.array(tokens_positive)
                    tokens_positive += len(output["text"]) + len(self.sep_token)
                    tokens_positive = tokens_positive.tolist()
                output["text"] += self.sep_token + data_info["text"]
            if tokens_positive is not None:
                output["tokens_positive"].append(tokens_positive)
            for k in ["target_id", "distractor_ids", "target", "anchors", "anchor_ids"]:
                if k not in data_info:
                    continue
                output[k].append(data_info[k])
        output["scan_id"] = data_infos[0]["scan_id"]
        return output

    def get_data_info_grounding(self, data_info):

        flags = {}
        if "distractor_ids" in data_info:
            flags["is_unique"] = len(data_info["distractor_ids"]) == 0
            flags["is_hard"] = len(data_info["distractor_ids"]) > 3
        if "text" in data_info:
            flags["is_view_dep"] = self._is_view_dep(data_info["text"])

        scan_id = data_info["scan_id"]
        scan_data = copy.deepcopy(self.scans[scan_id])
        data_info = [data_info]
        if self.num_text > 1:
            data_idx = self.scan_id_to_data_idx[scan_id]
            sample_idx = sample(
                len(data_idx),
                max(min(int(np.random.rand()*self.num_text), len(data_idx))-1, 1),
                fix_interval=False
            )
            for i in sample_idx:
                data_info.append(super().get_data_info(data_idx[i]))
        data_info = self.merge_grounding_data(data_info)
        scan_data["text"] = data_info["text"]

        if "ann_info" in scan_data and "target" in data_info:
            tokens_positive = []
            obj_idx = []
            for i, (target_name, id) in enumerate(
                zip(data_info["target"], data_info["target_id"])
            ):
                mask = np.logical_and(
                    scan_data["ann_info"]["bbox_id"] == id,
                    np.array(scan_data["ann_info"]["gt_names"]) == target_name
                )
                if np.sum(mask) != 1:
                    continue
                obj_idx.append(np.where(mask)[0][0])
                tokens_positive.append(data_info["tokens_positive"][i])
            obj_idx = np.array(obj_idx, dtype=np.int32)
            scan_data["ann_info"]["gt_bboxes_3d"] = scan_data["ann_info"]["gt_bboxes_3d"][obj_idx]
            scan_data["ann_info"]["gt_labels_3d"] = scan_data["ann_info"]["gt_labels_3d"][obj_idx]
            scan_data["ann_info"]["gt_names"] = [
                scan_data["ann_info"]["gt_names"][i] for i in obj_idx
            ]
            if "visible_instance_masks" in scan_data["ann_info"]:
                scan_data["ann_info"]["visible_instance_masks"] = [
                    visible_instance_mask[obj_idx]
                    for visible_instance_mask in scan_data["ann_info"]["visible_instance_masks"]
                ]
            scan_data["tokens_positive"] = tokens_positive
            scan_data["eval_ann_info"] = scan_data["ann_info"]
            scan_data["eval_ann_info"].update(flags)
        elif "tokens_positive" in data_info:
            scan_data["tokens_positive"] = data_info.get("tokens_positive")
        return scan_data

    @force_full_init
    def get_data_info(self, idx):
        data_info = super().get_data_info(idx)
        if self.mode == "detection":
            return data_info
        elif self.mode == "continuous":
            return self.get_data_info_continuous(data_info)
        elif self.mode == "grounding":
            return self.get_data_info_grounding(data_info)


def index(input, idx):
    if isinstance(idx, int):
        idx = [idx]
    output = []
    for i in idx:
        output.append(input[i])
    return output
