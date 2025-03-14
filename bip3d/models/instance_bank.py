import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from bip3d.registry import MODELS

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@MODELS.register_module()
class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_current_instance=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
        anchor_in_camera=True,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_current_instance = num_current_instance
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        if anchor_handler is not None:
            anchor_handler = MODELS.build(anchor_handler)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        if len(anchor.shape) == 3:  # for map
            anchor = anchor.reshape(anchor.shape[0], -1)
        self.num_anchor = min(len(anchor), num_anchor)
        self.anchor = anchor[:num_anchor]
        # self.anchor = nn.Parameter(
        #     torch.tensor(anchor, dtype=torch.float32),
        #     requires_grad=anchor_grad,
        # )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([1, self.embed_dims]),
            # torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.anchor_in_camera = anchor_in_camera
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0

    def bbox_transform(self, bbox, matrix):
        # bbox: bs, n, 9
        # matrix: bs, cam, 4, 4
        # output: bs, n*cam, 9
        bbox = bbox.unsqueeze(dim=2)
        matrix = matrix.unsqueeze(dim=1)
        points = bbox[..., :3]
        points_extend = torch.concat(
            [points, torch.ones_like(points[..., :1])], dim=-1
        )
        points_trans = torch.matmul(matrix, points_extend[..., None])[
            ..., :3, 0
        ]

        size = bbox[..., 3:6].tile(1, 1, points_trans.shape[2], 1)
        angle = bbox[..., 6:].tile(1, 1, points_trans.shape[2], 1)

        bbox = torch.cat([points_trans, size, angle], dim=-1)
        bbox = bbox.flatten(1, 2)
        return bbox

    def get(self, batch_size, metas=None, dn_metas=None):
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, self.anchor.shape[0], 1)
        )
        anchor = torch.tile(
            instance_feature.new_tensor(self.anchor)[None], (batch_size, 1, 1)
        )

        if self.anchor_in_camera:
            cam2global = np.linalg.inv(metas["extrinsic"].cpu().numpy())
            cam2global = torch.from_numpy(cam2global).to(anchor)
            anchor = self.bbox_transform(anchor, cam2global)
            instance_feature = instance_feature.tile(1, cam2global.shape[1], 1)

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            # assert False, "TODO: linxuewu"
            # history_time = self.metas["timestamp"]
            # time_interval = metas["timestamp"] - history_time
            # time_interval = time_interval.to(dtype=instance_feature.dtype)
            # self.mask = torch.abs(time_interval) <= self.max_time_interval

            last_scan_id = self.metas["scan_id"]
            current_scan_id = metas["scan_id"]
            self.mask = torch.tensor(
                [x==y for x,y in zip(last_scan_id, current_scan_id)],
                device=anchor.device,
            )
            assert self.mask.shape[0] == 1
            if not self.mask:
                self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence, num_dn=None):
        if self.cached_feature is None:
            return instance_feature, anchor

        if num_dn is not None and num_dn > 0:
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : -num_dn]
            anchor = anchor[:, : -num_dn]
            confidence = confidence[:, : -num_dn]

        N = self.num_current_instance
        if N is not None and N < confidence.shape[1]:
            confidence = confidence.max(dim=-1).values
            _, (selected_feature, selected_anchor) = topk(
                confidence, N, instance_feature, anchor
            )
        else:
            selected_feature, selected_anchor = instance_feature, anchor
        instance_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )

        if num_dn is not None and num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            N = self.confidence.shape[1]
            confidence[:, : N] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : N],
            )
        self.temp_confidence = confidence

        if self.num_temp_instances < confidence.shape[1]:
            (
                self.confidence,
                (self.cached_feature, self.cached_anchor),
            ) = topk(
                confidence, self.num_temp_instances, instance_feature, anchor
            )
        else:
            self.confidence, self.cached_feature, self.cached_anchor = (
                confidence, instance_feature, anchor
            )

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()

        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        instance_id = instance_id.squeeze(dim=-1)
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )
