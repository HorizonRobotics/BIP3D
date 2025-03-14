import os
import torch

def get_renamed_ckpt(file, output="./"):
    ckpt_rename = dict()
    ckpt = torch.load(file)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    for key,value in ckpt.items():
        k = None
        if key.startswith("backbone") or key.startswith("neck"):
            k = key
        elif key.startswith("language_model."):
            k = key.replace("language_model.", "text_encoder.")
        elif key.startswith("encoder."):
            if key.startswith("encoder.layers."):
                k = key.replace("encoder.layers.", "feature_enhancer.img_attn_blocks.")
            elif key.startswith("encoder.fusion_layers."):
                k = key.replace("encoder.fusion_layers.", "feature_enhancer.text_img_attn_blocks.")
            elif key.startswith("encoder.text_layers."):
                k = key.replace("encoder.text_layers.", "feature_enhancer.text_attn_blocks.")
        elif key.startswith("decoder.layers."):
            ops = [
                "gnn",
                "norm",
                "text_cross_attn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ]
            layer_id = int(key.split(".")[2])
            name = key.split(".")[3]
            for op in ops:
                if name == "self_attn":
                    block_id = 0
                elif name == "cross_attn_text":
                    block_id = 2
                elif name == "cross_attn":
                    block_id = 4
                elif name == "ffn":
                    block_id = 6
                elif name == "norms":
                    norm_id = int(key.split(".")[4])
                    block_id = (norm_id + 1) * 2 - 1
            op_id = block_id + layer_id * len(ops)
            k = f"decoder.layers.{op_id}."
            if name == "norms":
                k += ".".join(key.split(".")[5:])
            else:
                k += ".".join(key.split(".")[4:])
        elif key.startswith("bbox_head."):
            layer_id = int(key.split(".")[2])
            op_id = 8 + layer_id * 9
            if "reg_branches" in key:
                k = f"decoder.layers.{op_id}." + ".".join(key.split(".")[3:])
            elif "cls_branches" in key:
                k = f"decoder.layers.{op_id}.bias"
        elif "pts_prob_fc" in key:
            k = "spatial_enhancer." + key
        elif key in [
            "pts_prob_pre_fc.weight",
            "pts_prob_pre_fc.bias",
            "pts_fc.weight",
            "pts_fc.bias",
            "fusion_fc.0.layers.0.0.weight",
            "fusion_fc.0.layers.0.0.bias",
            "fusion_fc.0.layers.1.weight",
            "fusion_fc.0.layers.1.bias",
            "fusion_fc.1.weight",
            "fusion_fc.1.bias",
            "fusion_norm.weight",
            "fusion_norm.bias",
        ]:
            k = "spatial_enhancer." + key
        elif key == "level_embed":
            k = "feature_enhancer.level_embed"
        elif key == "query_embedding.weight":
            k = "decoder.instance_bank.instance_feature"
        if k is None:
    #         print(key)
            k = key
        if k == "decoder.norm.weight":
            print(key)
        ckpt_rename[k] = value

    path, file_name = os.path.split(file)
    file_name = file_name[:-4]+"-rename.pth"
    output_file = os.path.join(output, file_name)
    torch.save(ckpt_rename, output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='mmdet Grounding-DINO checkpoint rename to BIP3D')
    parser.add_argument("file")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.output is None:
        output = "./"
    get_renamed_ckpt(args.file, output)
