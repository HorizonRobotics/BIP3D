import torch
from torch import nn

from mmengine.model import BaseModel
from mmdet.models.layers.transformer.utils import MLP
from mmcv.cnn.bricks.transformer import FFN

from .utils import deformable_format
from bip3d.registry import MODELS


@MODELS.register_module()
class DepthFusionSpatialEnhancer(BaseModel):
    def __init__(
        self,
        embed_dims=256,
        feature_3d_dim=32,
        num_depth_layers=2,
        min_depth=0.25,
        max_depth=10,
        num_depth=64,
        with_feature_3d=True,
        loss_depth_weight=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.feature_3d_dim = feature_3d_dim
        self.num_depth_layers = num_depth_layers
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_depth = num_depth
        self.with_feature_3d = with_feature_3d
        self.loss_depth_weight = loss_depth_weight

        fusion_dim = self.embed_dims + self.feature_3d_dim
        if self.with_feature_3d:
            self.pts_prob_pre_fc = nn.Linear(
                self.embed_dims, self.feature_3d_dim
            )
            dim = self.feature_3d_dim * 2
            fusion_dim += self.feature_3d_dim
        else:
            dim = self.embed_dims
        self.pts_prob_fc = MLP(
            dim,
            dim,
            self.num_depth,
            self.num_depth_layers,
        )
        self.pts_fc = nn.Linear(3, self.feature_3d_dim)
        self.fusion_fc = nn.Sequential(
            FFN(embed_dims=fusion_dim, feedforward_channels=1024),
            nn.Linear(fusion_dim, self.embed_dims),
        )
        self.fusion_norm = nn.LayerNorm(self.embed_dims)

    def forward(
        self,
        feature_maps,
        feature_3d=None,
        batch_inputs=None,
        **kwargs,
    ):
        with_cams = feature_maps[0].dim() == 5
        if with_cams:
            bs, num_cams = feature_maps[0].shape[:2]
        else:
            bs = feature_maps[0].shape[0]
            num_cams = 1

        feature_2d, spatial_shapes, _ = deformable_format(feature_maps)
        pts = self.get_pts(
            spatial_shapes,
            batch_inputs["image_wh"][0, 0],
            batch_inputs["projection_mat"],
            feature_2d.device,
            feature_2d.dtype,
        )

        if self.with_feature_3d:
            feature_3d = deformable_format(feature_3d)[0]
            depth_prob_feat = self.pts_prob_pre_fc(feature_2d)
            depth_prob_feat = torch.cat([depth_prob_feat, feature_3d], dim=-1)
            depth_prob = self.pts_prob_fc(depth_prob_feat).softmax(dim=-1)
            feature_fused = [feature_2d, feature_3d]
        else:
            depth_prob = self.pts_prob_fc(feature_2d).softmax(dim=-1)
            feature_fused = [feature_2d]

        pts_feature = self.pts_fc(pts)
        pts_feature = (depth_prob.unsqueeze(dim=-1) * pts_feature).sum(dim=-2)
        feature_fused.append(pts_feature)
        feature_fused = torch.cat(feature_fused, dim=-1)
        feature_fused = self.fusion_fc(feature_fused) + feature_2d
        feature_fused = self.fusion_norm(feature_fused)
        feature_fused = deformable_format(feature_fused, spatial_shapes)
        if self.loss_depth_weight > 0 and self.training:
            loss_depth = self.depth_prob_loss(depth_prob, batch_inputs)
        else:
            loss_depth = None
        return feature_fused, depth_prob, loss_depth

    def get_pts(self, spatial_shapes, image_wh, projection_mat, device, dtype):
        pixels = []
        for i, shape in enumerate(spatial_shapes):
            stride = image_wh[0] / shape[1]
            u = torch.linspace(
                0, image_wh[0] - stride, shape[1], device=device, dtype=dtype
            )
            v = torch.linspace(
                0, image_wh[1] - stride, shape[0], device=device, dtype=dtype
            )
            u = u[None].tile(shape[0], 1)
            v = v[:, None].tile(1, shape[1])
            uv = torch.stack([u, v], dim=-1).flatten(0, 1)
            pixels.append(uv)
        pixels = torch.cat(pixels, dim=0)[:, None]
        depths = torch.linspace(
            self.min_depth,
            self.max_depth,
            self.num_depth,
            device=device,
            dtype=dtype,
        )
        depths = depths[None, :, None]
        pts = pixels * depths
        depths = depths.tile(pixels.shape[0], 1, 1)
        pts = torch.cat([pts, depths, torch.ones_like(depths)], dim=-1)

        pts = torch.linalg.solve(
            projection_mat.mT.unsqueeze(dim=2), pts, left=False
        )[
            ..., :3
        ]  # b,cam,N,3
        return pts

    def depth_prob_loss(self, depth_prob, batch_inputs):
        mask = batch_inputs["depth_prob_gt"][..., 0] != 1
        loss_depth = (
            torch.nn.functional.binary_cross_entropy(
                depth_prob[mask], batch_inputs["depth_prob_gt"][mask]
            )
            * self.loss_depth_weight
        )
        return loss_depth
