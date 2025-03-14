import torch
from torch.autograd.function import Function, once_differentiable
from torch.cuda.amp import custom_fwd, custom_bwd

from . import deformable_aggregation_ext as da
from . import deformable_aggregation_with_depth_ext as dad


class DeformableAggregationFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        # output: [bs, num_pts, num_embeds]
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = da.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        da.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        # print(grad_mc_ms_feat.abs().mean(), grad_sampling_location.abs().mean(), grad_weights.abs().mean())
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )

class DeformableAggregationWithDepthFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
        num_depths,
    ):
        # output: [bs, num_pts, num_embeds]
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = dad.deformable_aggregation_with_depth_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            num_depths,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx._num_depths = num_depths
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        num_depths = ctx._num_depths
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        dad.deformable_aggregation_with_depth_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            num_depths,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        # print(grad_mc_ms_feat.abs().mean(), grad_sampling_location.abs().mean(), grad_weights.abs().mean())
        # print(grad_mc_ms_feat.abs().max(), grad_sampling_location.abs().max(), grad_weights.abs().max())
        # print("")
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
            None,
        )


def deformable_aggregation_func(
    mc_ms_feat,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
    depth_prob=None,
    depth=None
):
    if depth_prob is not None and depth is not None:
        mc_ms_feat = torch.cat([mc_ms_feat, depth_prob], dim=-1)
        sampling_location = torch.cat([sampling_location, depth], dim=-1)
        return DeformableAggregationWithDepthFunction.apply(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            depth_prob.shape[-1],
        )
    else:
        return DeformableAggregationFunction.apply(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
 
def feature_maps_format(feature_maps, inverse=False):
    bs, num_cams = feature_maps[0].shape[:2]
    if not inverse:
        spatial_shape = []
        scale_start_index = [0]

        col_feats = []
        for i, feat in enumerate(feature_maps):
            spatial_shape.append(feat.shape[-2:])
            scale_start_index.append(
                feat.shape[-1] * feat.shape[-2] + scale_start_index[-1]
            )
            col_feats.append(torch.reshape(
                feat, (bs, num_cams, feat.shape[2], -1)
            ))
        scale_start_index.pop()
        col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2)
        feature_maps = [
            col_feats,
            torch.tensor(
                spatial_shape,
                dtype=torch.int64,
                device=col_feats.device,
            ),
            torch.tensor(
                scale_start_index,
                dtype=torch.int64,
                device=col_feats.device,
            ),
        ]
    else:
        spatial_shape = feature_maps[1].int()
        split_size = (spatial_shape[:, 0] * spatial_shape[:, 1]).tolist()
        feature_maps = feature_maps[0].permute(0, 1, 3, 2)
        feature_maps = list(torch.split(feature_maps, split_size, dim=-1))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = feat.reshape(
                feat.shape[:3] + (spatial_shape[i, 0], spatial_shape[i, 1])
            )
    return feature_maps
