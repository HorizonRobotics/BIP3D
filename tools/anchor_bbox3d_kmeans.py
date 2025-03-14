import torch
import pickle
import tqdm
from sklearn.cluster import KMeans
import numpy as np

from bip3d.structures.bbox_3d import EulerDepthInstance3DBoxes


def sample(ids, n):
    if n == len(ids):
        return np.copy(ids)
    elif n > len(ids):
        return np.concatenate(
            [ids, sample(ids, n - len(ids))]
        )
    else:
        interval = len(ids) / n
        output = []
        for i in range(n):
            output.append(ids[int(interval*i)])
        return np.array(output)


def kmeans(
    ann_file,
    output_file,
    z_min=-0.2,
    z_max=3,
):
    ann = pickle.load(open(ann_file, "rb"))
    all_cam_bbox = []
    for x in tqdm.tqdm(ann["data_list"]):
        bbox = np.array([y["bbox_3d"] for y in x["instances"]])
        ids = np.arange(len(x["images"]))
        ids = sample(ids, 50)
        for idx in ids:
            image = x["images"][idx]
            global2cam = np.linalg.inv(x['axis_align_matrix'] @ image['cam2global'])
            _bbox = EulerDepthInstance3DBoxes(np.copy(bbox[image["visible_instance_ids"]]))
            mask = torch.logical_and(
                _bbox.tensor[:, 2] > z_min,
                _bbox.tensor[:, 2] < z_max
            )
            _bbox = _bbox[mask]
            _bbox.transform(global2cam)
            all_cam_bbox.append(_bbox.tensor.numpy())
    all_cam_bbox = np.concatenate(all_cam_bbox)
    print("start to kmeans, please wait")
    cluster_cam = KMeans(n_clusters=100).fit(all_cam_bbox).cluster_centers_
    cluster_cam[:, 3:6] = np.log(cluster_cam[:, 3:6])
    np.save(output_file, cluster_cam)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='anchor bbox3d kmeans for embodiedscan dataset')
    parser.add_argument("ann_file")
    parser.add_argument("--output_file")
    parser.add_argument("--z_min", defaule=-0.2)
    parser.add_argument("--z_max", defaule=3)
    args = parser.parse_args()
    kmeans(args.ann_file, args.output_file, args.z_min, args.z_max)
