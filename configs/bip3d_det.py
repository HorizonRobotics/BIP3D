_base_ = ["./default_runtime.py"]

import os
from bip3d.datasets.embodiedscan_det_grounding_dataset import (
    class_names, head_labels, common_labels, tail_labels
)

DEBUG = os.environ.get("CLUSTER") is None
del os

backend_args = None

metainfo = dict(classes="all")

depth = True
depth_loss = True

z_range=[-0.2, 3]
min_depth = 0.25
max_depth = 10
num_depth = 64

model = dict(
    type="BIP3D",
    input_3d="depth_img",
    use_depth_grid_mask=True,
    data_preprocessor=dict(
        type="CustomDet3DDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="mmdet.SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=False,
    ),
    neck=dict(
        type="mmdet.ChannelMapper",
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    text_encoder=dict(
        type="BertModel",
        special_tokens_list=["[CLS]", "[SEP]"],
        name="./ckpt/bert-base-uncased",
        pad_to_max=False,
        use_sub_sentence_represent=True,
        add_pooling_layer=False,
        max_tokens=768,
        use_checkpoint=True,
        return_tokenized=True,
    ),
    backbone_3d=(
        dict(
            type="mmdet.ResNet",
            depth=34,
            in_channels=1,
            base_channels=4,
            num_stages=4,
            out_indices=(1, 2, 3),
            norm_cfg=dict(type="BN", requires_grad=True),
            norm_eval=True,
            with_cp=True,
            style="pytorch",
        )
        if depth
        else None
    ),
    neck_3d=(
        dict(
            type="mmdet.ChannelMapper",
            in_channels=[8, 16, 32],
            kernel_size=1,
            out_channels=32,
            act_cfg=None,
            bias=True,
            norm_cfg=dict(type="GN", num_groups=4),
            num_outs=4,
        )
        if depth
        else None
    ),
    feature_enhancer=dict(
        type="TextImageDeformable2DEnhancer",
        num_layers=6,
        text_img_attn_block=dict(
            v_dim=256, l_dim=256, embed_dim=1024, num_heads=4, init_values=1e-4
        ),
        img_attn_block=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_levels=4, dropout=0.0, im2col_step=64
            ),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0
            ),
        ),
        text_attn_block=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0
            ),
        ),
        embed_dims=256,
        num_feature_levels=4,
        positional_encoding=dict(
            num_feats=128, normalize=True, offset=0.0, temperature=20
        ),
    ),
    spatial_enhancer=dict(
        type="DepthFusionSpatialEnhancer",
        embed_dims=256,
        feature_3d_dim=32,
        num_depth_layers=2,
        min_depth=min_depth,
        max_depth=max_depth,
        num_depth=num_depth,
        with_feature_3d=depth,
        loss_depth_weight=1.0 if depth_loss else -1,
    ),
    decoder=dict(
        type="BBox3DDecoder",
        look_forward_twice=True,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=50,
            anchor="anchor_files/embodiedscan_kmeans.npy",
            embed_dims=256,
            anchor_in_camera=True,
        ),
        anchor_encoder=dict(
            type="DoF9BoxEncoder",
            embed_dims=256,
            rot_dims=3,
        ),
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=256,
            num_heads=8,
            dropout=0.0,
            batch_first=True,
        ),
        ffn=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=2048,
            ffn_drop=0.0,
        ),
        norm_layer=dict(type="LN", normalized_shape=256),
        deformable_model=dict(
            type="DeformableFeatureAggregation",
            embed_dims=256,
            num_groups=8,
            num_levels=4,
            use_camera_embed=True,
            use_deformable_func=True,
            with_depth=True,
            min_depth=min_depth,
            max_depth=max_depth,
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
                num_learnable_pts=9,
            ),
            with_value_proj=True,
            filter_outlier=True,
        ),
        text_cross_attn=dict(
            type="MultiheadAttention",
            embed_dims=256,
            num_heads=8,
            dropout=0.0,
            batch_first=True,
        ),
        refine_layer=dict(
            type="GroundingRefineClsHead",
            embed_dims=256,
            output_dim=9,
            cls_bias=True,
            # cls_layers=True,
        ),
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_reg=dict(
            type="DoF9BoxLoss",
            loss_weight_wd=1.0,
            loss_weight_pcd=0.0,
            loss_weight_cd=0.8,
        ),
        sampler=dict(
            type="Grounding3DTarget",
            cls_weight=1.0,
            box_weight=1.0,
            num_dn=100,
            cost_weight_wd=1.0,
            cost_weight_pcd=0.0,
            cost_weight_cd=0.8,
            with_dn_query=True,
            num_classes=284,
            embed_dims=256,
        ),
        gt_reg_key="gt_bboxes_3d",
        gt_cls_key="tokens_positive",
        post_processor=dict(
            type="GroundingBox3DPostProcess",
            num_output=1000,
        ),
        with_instance_id=False,
    ),
)

dataset_type = "EmbodiedScanDetGroundingDataset"
data_root = "data"
metainfo = dict(
    classes=class_names,
    classes_split=(head_labels, common_labels, tail_labels),
    box_type_3d="euler-depth",
)


image_wh = (512, 512)

rotate_3rscan = True
sep_token = "[SEP]"
cam_standardization = True
if cam_standardization:
    resize = dict(
        type="CamIntrisicStandardization",
        dst_intrinsic=[
            [432.57943431339237, 0.0, 256],
            [0.0, 539.8570854208559, 256],
            [0.0, 0.0, 1.0],
        ],
        dst_wh=image_wh,
    )
else:
    resize = dict(type="CustomResize", scale=image_wh, keep_ratio=False)

train_pipeline = [
    dict(type="LoadAnnotations3D"),
    dict(
        type="MultiViewPipeline",
        n_images=1,
        max_n_images=18,
        transforms=[
            dict(type="LoadImageFromFile", backend_args=backend_args),
            dict(type="LoadDepthFromFile", backend_args=backend_args),
            resize,
            dict(
                type="ResizeCropFlipImage",
                data_aug_conf={
                    "resize_lim": (1.0, 1.0),
                    "final_dim": image_wh,
                    "bot_pct_lim": (0.0, 0.05),
                    "rot_lim": (0, 0),
                    "H": image_wh[1],
                    "W": image_wh[0],
                    "rand_flip": False,
                },
            ),
        ],
        rotate_3rscan=rotate_3rscan,
        ordered=False,
    ),
    dict(
        type="CategoryGroundingDataPrepare",
        classes=class_names,
        filter_others=True,
        sep_token=sep_token,
        max_class=128,
        training=True,
        z_range=z_range,
    ),
    dict(
        type="Pack3DDetInputs",
        keys=["img", "depth_img", "gt_bboxes_3d", "gt_labels_3d"],
    ),
]
if depth_loss:
    train_pipeline.append(
        dict(
            type="DepthProbLabelGenerator",
            origin_stride=4,
            min_depth=min_depth,
            max_depth=max_depth,
            num_depth=num_depth,
        ),
    )

test_pipeline = [
    dict(type="LoadAnnotations3D"),
    dict(
        type="MultiViewPipeline",
        n_images=1,
        max_n_images=50,
        ordered=True,
        transforms=[
            dict(type="LoadImageFromFile", backend_args=backend_args),
            dict(type="LoadDepthFromFile", backend_args=backend_args),
            resize,
        ],
        rotate_3rscan=rotate_3rscan,
    ),
    dict(
        type="CategoryGroundingDataPrepare",
        classes=class_names,
        sep_token=sep_token,
        training=False,
        filter_others=False,
    ),
    dict(
        type="Pack3DDetInputs",
        keys=["img", "depth_img", "gt_bboxes_3d", "gt_labels_3d"],
    ),
]

trainval = False
data_version = "v1"

if data_version == "v1":
    train_ann_file = "embodiedscan/embodiedscan_infos_train.pkl"
    val_ann_file = "embodiedscan/embodiedscan_infos_val.pkl"
elif data_version == "v2":
    train_ann_file = "embodiedscan-v2/embodiedscan_infos_train.pkl"
    val_ann_file = "embodiedscan-v2/embodiedscan_infos_val.pkl"
else:
    assert False

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=train_ann_file,
    pipeline=train_pipeline,
    test_mode=False,
    filter_empty_gt=True,
    box_type_3d="Euler-Depth",
    metainfo=metainfo,
)


if trainval:
    train_dataset = dict(
        type="ConcatDataset",
        datasets=[
            train_dataset,
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=val_ann_file,
                pipeline=train_pipeline,
                test_mode=False,
                filter_empty_gt=True,
                box_type_3d="Euler-Depth",
                metainfo=metainfo,
            )
        ]
    )


train_dataloader = dict(
    batch_size=1,
    num_workers=4 if not DEBUG else 0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=10,
        dataset=train_dataset,
    ),
)

    
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        test_mode=True,
        filter_empty_gt=True,
        box_type_3d="Euler-Depth",
        metainfo=metainfo,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="IndoorDetMetric",
    collect_dir="/job_data/.dist_test" if not DEBUG else None,
    # collect_device="gpu"
)
test_evaluator = val_evaluator

max_epochs = 24
train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

lr = 2e-4
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.0005),
    paramwise_cfg=dict(
        custom_keys={
            "backbone.": dict(lr_mult=0.1),
            "text_encoder": dict(lr_mult=0.05),
            "absolute_pos_embed": dict(decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=10, norm_type=2),
)


# learning rate
param_scheduler = [
    dict(
        type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500
    ),
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[int(max_epochs / 12 * 8), int(max_epochs / 12 * 11)],
        gamma=0.1,
    ),
]

custom_hooks = [dict(type="EmptyCacheHook", after_iter=False)]
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3),
)

vis_backends = [
    dict(
        type="TensorboardVisBackend",
        save_dir="/job_tboard" if not DEBUG else "./work-dir",
    ),
]

visualizer = dict(
    type="Visualizer",
    vis_backends=vis_backends,
    name="visualizer",
)

load_from = "ckpt/groundingdino_swint_ogc_mmdet-822d7e9d-rename.pth"
