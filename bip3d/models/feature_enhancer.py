import torch
from torch import nn

from mmengine.model import BaseModel
from mmdet.models.layers.transformer.utils import get_text_sine_pos_embed
from mmdet.models.layers import SinePositionalEncoding
from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoder as DDTE,
)
from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoderLayer,
)
from mmdet.models.layers.transformer.detr_layers import (
    DetrTransformerEncoderLayer,
)
from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock
from bip3d.registry import MODELS

from .utils import deformable_format


@MODELS.register_module()
class TextImageDeformable2DEnhancer(BaseModel):
    def __init__(
        self,
        num_layers,
        text_img_attn_block,
        img_attn_block,
        text_attn_block=None,
        embed_dims=256,
        num_feature_levels=4,
        positional_encoding=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_feature_levels = num_feature_levels
        self.embed_dims = embed_dims
        self.positional_encoding = positional_encoding
        self.text_img_attn_blocks = nn.ModuleList()
        self.img_attn_blocks = nn.ModuleList()
        self.text_attn_blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.text_img_attn_blocks.append(
                SingleScaleBiAttentionBlock(**text_img_attn_block)
            )
            self.img_attn_blocks.append(
                DeformableDetrTransformerEncoderLayer(**img_attn_block)
            )
            self.text_attn_blocks.append(
                DetrTransformerEncoderLayer(**text_attn_block)
            )
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding
        )
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )

    def forward(
        self,
        feature_maps,
        text_dict=None,
        **kwargs,
    ):
        with_cams = feature_maps[0].dim() == 5
        if with_cams:
            bs, num_cams = feature_maps[0].shape[:2]
            feature_maps = [x.flatten(0, 1) for x in feature_maps]
        else:
            bs = feature_maps[0].shape[0]
            num_cams = 1
        pos_2d = self.get_2d_position_embed(feature_maps)
        feature_2d, spatial_shapes, level_start_index = deformable_format(
            feature_maps
        )

        reference_points = DDTE.get_encoder_reference_points(
            spatial_shapes,
            valid_ratios=feature_2d.new_ones(
                [bs * num_cams, self.num_feature_levels, 2]
            ),
            device=feature_2d.device,
        )

        text_feature = text_dict["embedded"]
        pos_text = get_text_sine_pos_embed(
            text_dict["position_ids"][..., None],
            num_pos_feats=self.embed_dims,
            exchange_xy=False,
        )

        for layer_id in range(self.num_layers):
            feature_2d_fused = feature_2d[:, level_start_index[-1] :]
            if with_cams:
                feature_2d_fused = feature_2d_fused.unflatten(
                    0, (bs, num_cams)
                )
                feature_2d_fused = feature_2d_fused.flatten(1, 2)
            feature_2d_fused, text_feature = self.text_img_attn_blocks[
                layer_id
            ](
                feature_2d_fused,
                text_feature,
                attention_mask_l=text_dict["text_token_mask"],
            )
            if with_cams:
                feature_2d_fused = feature_2d_fused.unflatten(
                    1, (num_cams, -1)
                )
                feature_2d_fused = feature_2d_fused.flatten(0, 1)
            feature_2d = torch.cat(
                [feature_2d[:, : level_start_index[-1]], feature_2d_fused],
                dim=1,
            )

            feature_2d = self.img_attn_blocks[layer_id](
                query=feature_2d,
                query_pos=pos_2d,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=None,
            )

            text_attn_mask = text_dict.get("masks")
            if text_attn_mask is not None:
                text_num_heads = self.text_attn_blocks[
                    layer_id
                ].self_attn_cfg.num_heads
                text_attn_mask = ~text_attn_mask.repeat(text_num_heads, 1, 1)
            text_feature = self.text_attn_blocks[layer_id](
                query=text_feature,
                query_pos=pos_text,
                attn_mask=text_attn_mask,
                key_padding_mask=None,
            )
        feature_2d = deformable_format(
            feature_2d, spatial_shapes, batch_size=bs if with_cams else None
        )
        return feature_2d, text_feature

    def get_2d_position_embed(self, feature_maps):
        pos_2d = []
        for lvl, feat in enumerate(feature_maps):
            batch_size, c, h, w = feat.shape
            pos = self.positional_encoding(None, feat)
            pos = pos.view(batch_size, c, h * w).permute(0, 2, 1)
            pos = pos + self.level_embed[lvl]
            pos_2d.append(pos)
        pos_2d = torch.cat(pos_2d, 1)
        return pos_2d
