import torch
from torch import nn
from pytorch3d.transforms import euler_angles_to_matrix


def deformable_format(
    feature_maps,
    spatial_shapes=None,
    level_start_index=None,
    flat_batch=False,
    batch_size=None,
):
    if spatial_shapes is None:
        if flat_batch and feature_maps[0].dim() > 4:
            feature_maps = [x.flatten(end_dim=-4) for x in feature_maps]
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(feature_maps):
            spatial_shape = torch._shape_as_tensor(feat)[-2:].to(feat.device)
            feat = feat.flatten(start_dim=-2).transpose(-1, -2)
            feat_flatten.append(feat)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, -2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),  # (num_level)
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        return feat_flatten, spatial_shapes, level_start_index
    else:
        split_size = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        feature_maps = feature_maps.transpose(-1, -2)
        feature_maps = list(torch.split(feature_maps, split_size, dim=-1))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = feature_maps[i].unflatten(
                -1, (spatial_shapes[i, 0], spatial_shapes[i, 1])
            )
            if batch_size is not None:
                if isinstance(batch_size, int):
                    feature_maps[i] = feature_maps[i].unflatten(
                        0, (batch_size, -1)
                    )
                else:
                    feature_maps[i] = feature_maps[i].unflatten(
                        0, batch_size + (-1,)
                    )
        return feature_maps


def bbox_to_corners(bbox, permutation=False):
    assert (
        len(bbox.shape) == 2
    ), "bbox must be 2D tensor of shape (N, 6) or (N, 7) or (N, 9)"
    if bbox.shape[-1] == 6:
        rot_mat = (
            torch.eye(3, device=bbox.device)
            .unsqueeze(0)
            .repeat(bbox.shape[0], 1, 1)
        )
    elif bbox.shape[-1] == 7:
        angles = bbox[:, 6:]
        fake_angles = torch.zeros_like(angles).repeat(1, 2)
        angles = torch.cat((angles, fake_angles), dim=1)
        rot_mat = euler_angles_to_matrix(angles, "ZXY")
    elif bbox.shape[-1] == 9:
        rot_mat = euler_angles_to_matrix(bbox[:, 6:], "ZXY")
    else:
        raise NotImplementedError
    centers = bbox[:, :3].unsqueeze(1).repeat(1, 8, 1)  # shape (N, 8, 3)
    half_sizes = (
        bbox[:, 3:6].unsqueeze(1).repeat(1, 8, 1) / 2
    )  # shape (N, 8, 3)
    eight_corners_x = (
        torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], device=bbox.device)
        .unsqueeze(0)
        .repeat(bbox.shape[0], 1)
    )  # shape (N, 8)
    eight_corners_y = (
        torch.tensor([1, 1, -1, -1, 1, 1, -1, -1], device=bbox.device)
        .unsqueeze(0)
        .repeat(bbox.shape[0], 1)
    )  # shape (N, 8)
    eight_corners_z = (
        torch.tensor([1, -1, 1, -1, 1, -1, 1, -1], device=bbox.device)
        .unsqueeze(0)
        .repeat(bbox.shape[0], 1)
    )  # shape (N, 8)
    eight_corners = torch.stack(
        (eight_corners_x, eight_corners_y, eight_corners_z), dim=-1
    )  # shape (N, 8, 3)
    eight_corners = eight_corners * half_sizes  # shape (N, 8, 3)
    # rot_mat: (N, 3, 3), eight_corners: (N, 8, 3)
    rotated_corners = torch.matmul(
        eight_corners, rot_mat.transpose(1, 2)
    )  # shape (N, 8, 3)
    corners = rotated_corners + centers

    if permutation:
        corners = corners[:, PERMUTE_INDEX]
    return corners


def wasserstein_distance(source, target):
    rot_mat_src = euler_angles_to_matrix(source[..., 6:9], "ZXY")
    sqrt_sigma_src = rot_mat_src @ (
        source[..., 3:6, None] * rot_mat_src.transpose(-2, -1)
    )

    rot_mat_tgt = euler_angles_to_matrix(target[..., 6:9], "ZXY")
    sqrt_sigma_tgt = rot_mat_tgt @ (
        target[..., 3:6, None] * rot_mat_tgt.transpose(-2, -1)
    )

    sigma_distance = sqrt_sigma_src - sqrt_sigma_tgt
    sigma_distance = sigma_distance.pow(2).sum(dim=-1).sum(dim=-1)
    center_distance = ((source[..., :3] - target[..., :3]) ** 2).sum(dim=-1)
    distance = sigma_distance + center_distance
    distance = distance.clamp(1e-7).sqrt()
    return distance


def permutation_corner_distance(source, target):
    source_corners = bbox_to_corners(source)  # N, 8, 3
    target_corners = bbox_to_corners(target, True)  # N, 48, 8, 3
    distance = torch.norm(
        source_corners.unsqueeze(dim=-2) - target_corners, p=2, dim=-1
    )
    distance = distance.mean(dim=-1).min(dim=-1).values
    return distance


def center_distance(source, target):
    return torch.norm(source[..., :3] - target[..., :3], p=2, dim=-1)


def get_positive_map(char_positive, text_dict):
    bs, text_length = text_dict["embedded"].shape[:2]
    tokenized = text_dict["tokenized"]
    positive_maps = []
    for i in range(bs):
        num_target = len(char_positive[i])
        positive_map = torch.zeros(
            (num_target, text_length), dtype=torch.float
        )
        for j, tok_list in enumerate(char_positive[i]):
            for beg, end in tok_list:
                try:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
                except Exception as e:
                    print("beg:", beg, "end:", end)
                    print("token_positive:", char_positive[i])
                    raise e
                if beg_pos is None:
                    try:
                        beg_pos = tokenized.char_to_token(i, beg + 1)
                        if beg_pos is None:
                            beg_pos = tokenized.char_to_token(i, beg + 2)
                    except Exception:
                        beg_pos = None
                if end_pos is None:
                    try:
                        end_pos = tokenized.char_to_token(i, end - 2)
                        if end_pos is None:
                            end_pos = tokenized.char_to_token(i, end - 3)
                    except Exception:
                        end_pos = None
                if beg_pos is None or end_pos is None:
                    continue

                assert beg_pos is not None and end_pos is not None
                positive_map[j, beg_pos : end_pos + 1].fill_(1)
        positive_map /= (positive_map.sum(-1)[:, None] + 1e-6)
        positive_maps.append(positive_map)
    return positive_maps


def get_entities(text, char_positive, sep_token="[SEP]"):
    batch_entities = []
    for bs_idx in range(len(char_positive)):
        entities = []
        for obj_idx in range(len(char_positive[bs_idx])):
            entity = ""
            for beg, end in char_positive[bs_idx][obj_idx]:
                if len(entity) == 0:
                    entity = text[bs_idx][beg:end]
                else:
                    entity += sep_token + text[bs_idx][beg:end]
            entities.append(entity)
        batch_entities.append(entities)
    return batch_entities


def linear_act_ln(
    embed_dims, in_loops, out_loops, input_dims=None,
    act_cfg=None,
):
    if act_cfg is None:
        act_cfg = dict(type='ReLU', inplace=True)
    from mmcv.cnn import build_activation_layer
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(build_activation_layer(act_cfg))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers
