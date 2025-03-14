import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.cuda.amp.autocast_mode import autocast

from mmcv.cnn import Linear, Scale
from mmcv.ops import nms3d_normal, nms3d
from mmengine.model import BaseModel
from mmengine.structures import InstanceData
from mmdet.models.layers import SinePositionalEncoding
from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoder as DDTE,
)
from mmdet.models.layers.transformer.utils import MLP

from mmdet.utils import reduce_mean
from mmdet.models.detectors.glip import create_positive_map_label_to_token
from mmdet.models.dense_heads.atss_vlfusion_head import (
    convert_grounding_to_cls_scores,
)

from bip3d.registry import MODELS, TASK_UTILS
from bip3d.structures.bbox_3d.utils import rotation_3d_in_euler
from .utils import (
    deformable_format,
    wasserstein_distance,
    permutation_corner_distance,
    center_distance,
    get_positive_map,
    get_entities,
    linear_act_ln,
)

__all__ = ["BBox3DDecoder"]


X, Y, Z, W, L, H, ALPHA, BETA, GAMMA = range(9)


def decode_box(box, min_size=None, max_size=None):
    size = box[..., 3:6].exp()
    # size = box[..., 3:6]
    if min_size is not None or max_size is not None:
        size = size.clamp(min=min_size, max=max_size)
    box = torch.cat(
        [box[..., :3], size, box[..., 6:]],
        dim=-1,
    )
    return box


@MODELS.register_module()
class BBox3DDecoder(BaseModel):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        text_cross_attn: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        post_processor: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        gt_id_key: str = "instance_id",
        with_instance_id: bool = True,
        task_prefix: str = "det",
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        look_forward_twice: bool = False,
        init_cfg: dict = None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.gt_id_key = gt_id_key
        self.with_instance_id = with_instance_id
        self.task_prefix = task_prefix
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.look_forward_twice = look_forward_twice

        if reg_weights is None:
            self.reg_weights = [1.0] * 9
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "gnn",
                "norm",
                "text_cross_attn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry=MODELS):
            if cfg is None:
                return None
            return registry.build(cfg)

        self.instance_bank = build(instance_bank)
        self.anchor_encoder = build(anchor_encoder)
        self.sampler = build(sampler, TASK_UTILS)
        self.post_processor = build(post_processor, TASK_UTILS)
        self.loss_cls = build(loss_cls)
        self.loss_reg = build(loss_reg)
        self.op_config_map = {
            "temp_gnn": temp_graph_model,
            "gnn": graph_model,
            "norm": norm_layer,
            "ffn": ffn,
            "deformable": deformable_model,
            "text_cross_attn": text_cross_attn,
            "refine": refine_layer,
        }
        self.layers = nn.ModuleList(
            [
                build(self.op_config_map.get(op, None))
                for op in self.operation_order
            ]
        )
        self.embed_dims = self.instance_bank.embed_dims
        self.norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self):
        from mmengine.model import constant_init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for i, op in enumerate(self.operation_order):
            if op == "refine":
                m = self.layers[i]
                constant_init(m.layers[-2], 0, bias=0)
                constant_init(m.layers[-1], 1)
                nn.init.constant_(m.layers[-2].bias.data[2:], 0.0)

    def forward(
        self,
        feature_maps,
        text_dict=None,
        batch_inputs=None,
        depth_prob=None,
        **kwargs,
    ):
        batch_size = feature_maps[0].shape[0]
        feature_maps = list(deformable_format(feature_maps))

        # ========= get instance info ============
        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(
            batch_size, batch_inputs, dn_metas=self.sampler.dn_metas
        )

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            dn_metas = self.sampler.get_dn_anchors(
                batch_inputs[self.gt_cls_key],
                batch_inputs[self.gt_reg_key],
                text_dict=text_dict,
                label=batch_inputs["gt_labels_3d"],
            )
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_query,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            if dn_query is None:
                dn_query = instance_feature.new_zeros(
                    batch_size, num_dn_anchor, instance_feature.shape[-1]
                ),
            instance_feature = torch.cat(
                [instance_feature, dn_query], dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask
        else:
            num_dn_anchor = None
            num_free_instance = None

        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        # =================== forward the layers ====================
        prediction = []
        classification = []
        quality = []
        _anchor = None
        for i, (op, layer) in enumerate(
            zip(self.operation_order, self.layers)
        ):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = layer(
                    query=instance_feature,
                    key=temp_instance_feature,
                    value=temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=(
                        attn_mask if temp_instance_feature is None else None
                    ),
                )
            elif op == "gnn":
                instance_feature = layer(
                    query=instance_feature,
                    key=instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = layer(instance_feature)
            elif op == "deformable":
                instance_feature = layer(
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    batch_inputs,
                    depth_prob=depth_prob,
                )
            elif op == "text_cross_attn":
                text_feature = text_dict["embedded"]
                instance_feature = layer(
                    query=instance_feature,
                    key=text_feature,
                    value=text_feature,
                    query_pos=anchor_embed,
                    key_padding_mask=~text_dict["text_token_mask"],
                    key_pos=0,
                )
            elif op == "refine":
                _instance_feature = self.norm(instance_feature)
                if self.look_forward_twice:
                    if _anchor is None:
                        _anchor = anchor.clone()
                    _anchor, cls, qt = layer(
                        _instance_feature,
                        _anchor,
                        anchor_embed,
                        time_interval=time_interval,
                        text_feature=text_feature,
                        text_token_mask=text_dict["text_token_mask"],
                    )
                    prediction.append(_anchor)
                    anchor = layer(
                        instance_feature,
                        anchor,
                        anchor_embed,
                        time_interval=time_interval,
                    )[0]
                    anchor_embed = self.anchor_encoder(anchor)
                    _anchor = anchor
                    anchor = anchor.detach()
                else:
                    anchor, cls, qt = layer(
                        _instance_feature,
                        anchor,
                        anchor_embed,
                        time_interval=time_interval,
                        text_feature=text_feature,
                        text_token_mask=text_dict["text_token_mask"],
                    )
                    anchor_embed = self.anchor_encoder(anchor)
                    prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)

                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor if _anchor is None else _anchor,
                        cls, num_dn_anchor
                    )
                    anchor_embed = self.anchor_encoder(anchor)
                    if self.look_forward_twice:
                        _anchor = anchor
                        anchor = anchor.detach()
                    if dn_metas is not None:
                        num_instance = instance_feature.shape[1]
                        attn_mask = anchor.new_ones(
                            (num_instance, num_instance), dtype=torch.bool
                        )
                        attn_mask[:-num_dn_anchor, :-num_dn_anchor] = False
                        attn_mask[-num_dn_anchor:, -num_dn_anchor:] = dn_attn_mask

                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            dn_classification = [
                x[:, -num_dn_anchor:] for x in classification
            ]
            classification = [x[:, :-num_dn_anchor] for x in classification]
            dn_prediction = [x[:, -num_dn_anchor:] for x in prediction]
            prediction = [x[:, :-num_dn_anchor] for x in prediction]
            quality = [
                x[:, :-num_dn_anchor] if x is not None else None
                for x in quality
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        # "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, -num_dn_anchor:]
            dn_anchor = anchor[:, -num_dn_anchor:]
            instance_feature = instance_feature[:, :-num_dn_anchor]
            anchor_embed = anchor_embed[:, :-num_dn_anchor]
            anchor = anchor[:, :-num_dn_anchor]
            cls = cls[:, :-num_dn_anchor]

        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
                "instance_feature": instance_feature,
                "anchor_embed": anchor_embed,
            }
        )

        # cache current instances for temporal modeling
        self.instance_bank.cache(
            instance_feature, anchor, cls, batch_inputs, feature_maps
        )
        if self.with_instance_id:
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id
        return output

    def loss(self, model_outs, data, text_dict=None):
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            reg = decode_box(reg)
            cls_target, reg_target, reg_weights, ignore_mask = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
                text_dict=text_dict,
                ignore_mask=data.get("ignore_mask"),
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            reg_target_full = reg_target.clone()
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask = mask.reshape(-1)
            if ignore_mask is not None:
                ignore_mask = ~ignore_mask.reshape(-1)
                mask = torch.logical_and(mask, ignore_mask)
                ignore_mask = ignore_mask.tile(1, cls.shape[-1])
                
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            token_mask = torch.logical_not(cls.isinf())
            cls = cls[token_mask]
            cls_target = cls_target[token_mask]

            if ignore_mask is None:
                cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)
            else:
                ignore_mask = ignore_mask[token_mask]
                cls_loss = self.loss_cls(
                    cls[ignore_mask],
                    cls_target[ignore_mask],
                    avg_factor=num_pos,
                    weight=cls_mask[ignore_mask],
                )

            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            # cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                prefix=f"{self.task_prefix}_",
                suffix=f"_{decoder_idx}",
                quality=qt,
                # cls_target=cls_target,
            )

            output[f"{self.task_prefix}_loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        if "dn_prediction" not in model_outs:
            return output

        # ===================== denoising losses ======================
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)

        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls = cls.flatten(end_dim=1)[dn_valid_mask]
            mask = torch.logical_not(cls.isinf())
            cls_loss = self.loss_cls(
                cls[mask],
                dn_cls_target[mask],
                avg_factor=num_dn_pos,
            )

            reg = reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                ..., : len(self.reg_weights)
            ]
            reg = decode_box(reg)
            reg_loss = self.loss_reg(
                reg,
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                prefix=f"{self.task_prefix}_",
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"{self.task_prefix}_loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target.sum(dim=-1) > 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    def post_process(
        self,
        model_outs,
        text_dict,
        batch_inputs,
        batch_data_samples,
        results_in_data_samples=True,
    ):
        results = self.post_processor(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            text_dict=text_dict,
            batch_inputs=batch_inputs,
        )
        if results_in_data_samples:
            for i, ret in enumerate(results):
                instances = InstanceData()
                for k, v in ret.items():
                    if k == "bboxes_3d":
                        type = batch_data_samples[i].metainfo["box_type_3d"]
                        v = type(
                            v,
                            box_dim=v.shape[1],
                            origin=(0.5, 0.5, 0.5),
                        )
                    instances.__setattr__(k, v)
                batch_data_samples[i].pred_instances_3d = instances
            return batch_data_samples
        return results


@MODELS.register_module()
class GroundingRefineClsHead(BaseModel):
    def __init__(
        self,
        embed_dims=256,
        output_dim=9,
        scale=None,
        cls_layers=False,
        cls_bias=True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.refine_state = list(range(output_dim))
        self.scale = scale
        self.layers = nn.Sequential(
            *linear_act_ln(embed_dims, 2, 2),
            # MLP(embed_dims, embed_dims, embed_dims, 2),
            # nn.LayerNorm(self.embed_dims),
            # MLP(embed_dims, embed_dims, embed_dims, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )
        if cls_layers:
            self.cls_layers = nn.Sequential(
                MLP(embed_dims, embed_dims, embed_dims, 2),
                nn.LayerNorm(self.embed_dims),
            )
        else:
            self.cls_layers = nn.Identity()
        if cls_bias:
            bias_value = -math.log((1 - 0.01) / 0.01)
            self.bias = nn.Parameter(
                torch.Tensor([bias_value]), requires_grad=True
            )
        else:
            self.bias = None

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor = None,
        anchor_embed: torch.Tensor = None,
        time_interval: torch.Tensor = 1.0,
        text_feature=None,
        text_token_mask=None,
        **kwargs,
    ):
        if anchor_embed is not None:
            feature = instance_feature + anchor_embed
        else:
            feature = instance_feature
        output = self.layers(feature)
        if self.scale is not None:
            output = output * output.new_tensor(self.scale)
        if anchor is not None:
            output = output + anchor

        if text_feature is not None:
            cls = self.cls_layers(
                instance_feature) @ text_feature.transpose(-1, -2)
            cls = cls / math.sqrt(instance_feature.shape[-1])
            if self.bias is not None:
                cls = cls + self.bias
            if text_token_mask is not None:
                cls.masked_fill_(~text_token_mask[:, None, :], float("-inf"))
        else:
            cls = None
        return output, cls, None


@MODELS.register_module()
class DoF9BoxLoss(nn.Module):
    def __init__(
        self,
        loss_weight_wd=1.0,
        loss_weight_pcd=0.0,
        loss_weight_cd=0.8,
        decode_pred=False,
    ):
        super().__init__()
        self.loss_weight_wd = loss_weight_wd
        self.loss_weight_pcd = loss_weight_pcd
        self.loss_weight_cd = loss_weight_cd
        self.decode_pred = decode_pred

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        prefix="",
        suffix="",
        **kwargs,
    ):
        if box_target.shape[0] == 0:
            loss = box.sum() * 0
            return {f"{prefix}loss_box{suffix}": loss}
        if self.decode_pred:
            box = decode_box(box)
        loss = 0
        if self.loss_weight_wd > 0:
            loss += self.loss_weight_wd * wasserstein_distance(box, box_target)
        if self.loss_weight_pcd > 0:
            loss += self.loss_weight_pcd * permutation_corner_distance(
                box, box_target
            )
        if self.loss_weight_cd > 0:
            loss += self.loss_weight_cd * center_distance(box, box_target)

        if avg_factor is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / avg_factor
        output = {f"{prefix}loss_box{suffix}": loss}
        return output


@TASK_UTILS.register_module()
class GroundingBox3DPostProcess:
    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
        sorted: bool = True,
    ):
        super(GroundingBox3DPostProcess, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.sorted = sorted

    def __call__(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        quality=None,
        output_idx=-1,
        text_dict=None,
        batch_inputs=None,
    ):
        cls_scores = cls_scores[output_idx].sigmoid()
        if "tokens_positive" in batch_inputs:
            tokens_positive_maps = get_positive_map(
                batch_inputs["tokens_positive"],
                text_dict,
            )
            label_to_token = [
                create_positive_map_label_to_token(x, plus=1)
                for x in tokens_positive_maps
            ]
            cls_scores = convert_grounding_to_cls_scores(
                cls_scores, label_to_token
            )
            entities = get_entities(
                batch_inputs["text"],
                batch_inputs["tokens_positive"],
            )
        else:
            cls_scores, _ = cls_scores.max(dim=-1, keepdim=True)
            entities = batch_inputs["text"]

        # if squeeze_cls:
        #     cls_scores, cls_ids = cls_scores.max(dim=-1)
        #     cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        num_output = min(self.num_output, num_pred*num_cls)
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            num_output, dim=1, sorted=self.sorted
        )
        # if not squeeze_cls:
        cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if quality[output_idx] is None:
            quality = None
        if quality is not None:
            centerness = quality[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            # if not squeeze_cls:
            cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            # if squeeze_cls:
            #     category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]

            # nms_idx = nms3d(
            #     box[..., :7],
            #     scores,
            #     iou_threshold=0.4
            # )
            # box = box[nms_idx]
            # scores = scores[nms_idx]
            # category_ids = category_ids[nms_idx]

            if quality is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = decode_box(box, 0.1, 20)
            category_ids = category_ids.cpu()

            label_names = []
            for id in category_ids.tolist():
                if isinstance(entities[i], (tuple, list)):
                    label_names.append(entities[i][id])
                else:
                    label_names.append(entities[i])

            output.append(
                {
                    "bboxes_3d": box.cpu(),
                    "scores_3d": scores.cpu(),
                    "labels_3d": category_ids,
                    "target_scores_3d": scores.cpu(),
                    "label_names": label_names,
                }
            )
            if quality is not None:
                output[-1]["cls_scores"] = scores_origin.cpu()
            if instance_id is not None:
                ids = instance_id[i, indices[i]]
                if self.score_threshold is not None:
                    ids = ids[mask[i]]
                output[-1]["instance_ids"] = ids
        return output


@MODELS.register_module()
class DoF9BoxEncoder(nn.Module):
    def __init__(
        self,
        embed_dims,
        rot_dims=3,
        output_fc=True,
        in_loops=1,
        out_loops=2,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        def embedding_layer(input_dims, output_dims):
            return nn.Sequential(
                *linear_act_ln(output_dims, in_loops, out_loops, input_dims)
            )

        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        self.pos_fc = embedding_layer(3, embed_dims[0])
        self.size_fc = embedding_layer(3, embed_dims[1])
        self.yaw_fc = embedding_layer(rot_dims, embed_dims[2])
        self.rot_dims = rot_dims
        if output_fc:
            self.output_fc = embedding_layer(embed_dims[-1], embed_dims[-1])
        else:
            self.output_fc = None

    def forward(self, box_3d: torch.Tensor):
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        if box_3d.shape[-1] == 3:
            return pos_feat
        size_feat = self.size_fc(box_3d[..., [W, L, H]])
        yaw_feat = self.yaw_fc(box_3d[..., ALPHA : ALPHA + self.rot_dims])
        output = pos_feat + size_feat + yaw_feat
        if self.output_fc is not None:
            output = self.output_fc(output)
        return output


@MODELS.register_module()
class SparseBox3DKeyPointsGenerator(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = torch.tensor(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        bs, num_anchor = anchor.shape[:2]
        size = anchor[..., None, [W, L, H]].exp()
        key_points = self.fix_scale.to(anchor) * size
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            key_points = torch.cat(
                [key_points, learnable_scale * size], dim=-2
            )

        key_points = rotation_3d_in_euler(
            key_points.flatten(0, 1),
            anchor[..., [ALPHA, BETA, GAMMA]].flatten(0, 1),
        ).unflatten(0, (bs, num_anchor))
        key_points = key_points + anchor[..., None, [X, Y, Z]]

        if (
            cur_timestamp is None
            or temp_timestamps is None
            or len(temp_timestamps) == 0
        ) and T_cur2temp_list is None:
            return key_points

        temp_key_points_list = []
        velocity = anchor[..., VX:]
        for i, t_time in enumerate(temp_timestamps):
            time_interval = cur_timestamp - t_time
            translation = (
                velocity
                * time_interval.to(dtype=velocity.dtype)[:, None, None]
            )
            temp_key_points = key_points - translation[:, :, None]
            if T_cur2temp_list is not None:
                T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
                temp_key_points = T_cur2temp[:, None, None, :3] @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list

    @staticmethod
    def anchor_projection(
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            vel = anchor[..., VX:]
            vel_dim = vel.shape[-1]
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            center = anchor[..., [X, Y, Z]]
            if time_intervals is not None:
                time_interval = time_intervals[i]
            elif src_timestamp is not None and dst_timestamps is not None:
                time_interval = (src_timestamp - dst_timestamps[i]).to(
                    dtype=vel.dtype
                )
            else:
                time_interval = None
            if time_interval is not None:
                translation = vel.transpose(0, -1) * time_interval
                translation = translation.transpose(0, -1)
                center = center - translation
            center = (
                torch.matmul(
                    T_src2dst[..., :3, :3], center[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :3, 3]
            )
            size = anchor[..., [W, L, H]]
            yaw = torch.matmul(
                T_src2dst[..., :2, :2],
                anchor[..., [COS_YAW, SIN_YAW], None],
            ).squeeze(-1)
            yaw = yaw[..., [1, 0]]
            vel = torch.matmul(
                T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]
            ).squeeze(-1)
            dst_anchor = torch.cat([center, size, yaw, vel], dim=-1)
            dst_anchors.append(dst_anchor)
        return dst_anchors

    @staticmethod
    def distance(anchor):
        return torch.norm(anchor[..., :2], p=2, dim=-1)
