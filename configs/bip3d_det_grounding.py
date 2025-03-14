_base_ = ["./bip3d_grounding.py"]
from mmengine import Config
import os
det_config = Config.fromfile("configs/bip3d_det.py")
det_train_dataset = det_config.train_dataset
del Config, det_config

train_dataloader = _base_.train_dataloader
DEBUG = os.environ.get("CLUSTER") is None

train_dataloader["dataset"] = dict(
    type="ConcatDataset",
    datasets=[
        dict(
            type="RepeatDataset",
            times=20,
            dataset=det_train_dataset,
        ),
        train_dataloader["dataset"],
    ]
)


max_iters = 50000
train_cfg = dict(
    type="IterBasedTrainLoop", max_iters=max_iters, val_interval=25000,
    _delete_=True,
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
        end=max_iters,
        by_epoch=False,
        milestones=[int(max_iters*0.8), int(max_iters*0.95)],
        gamma=0.1,
    ),
]

custom_hooks = [dict(type="EmptyCacheHook", after_iter=False)]
default_hooks = dict(
    logger=dict(
        type="LoggerHook",
        interval=25,
        log_metric_by_epoch=False,
    ),
    checkpoint=dict(
        type="CheckpointHook", interval=25000, max_keep_ckpts=3, by_epoch=False
    ),
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
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=False)

load_from = "ckpt/bip3d_det.pth"
