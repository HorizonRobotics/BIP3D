_base_ = ["./bip3d_grounding.py"]

model = dict(
    backbone_3d=None,
    neck_3d=None,
    spatial_enhancer=dict(with_feature_3d=False),
    decoder=dict(deformable_model=dict(with_depth=False)),
)

load_from = "ckpt/bip3d_det_rgb.pth"
