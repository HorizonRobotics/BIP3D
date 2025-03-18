# BIP3D: Bridging 2D Images and 3D Perception for Embodied Intelligence

<div align="center" class="authors">
    <a href="https://scholar.google.com/citations?user=pfXQwcQAAAAJ&hl=en" target="_blank">Xuewu Lin</a>,
    <a href="https://wzmsltw.github.io/" target="_blank">Tianwei Lin</a>,
    <a href="https://scholar.google.com/citations?user=F2e_jZMAAAAJ&hl=en" target="_blank">Lichao Huang</a>,
    <a href="https://openreview.net/profile?id=~HONGYU_XIE2" target="_blank">Hongyu Xie</a>,
    <a href="https://scholar.google.com/citations?user=HQfc8TEAAAAJ&hl=en" target="_blank">Zhizhong Su</a>
</div>


<div align="center" style="line-height: 3;">
  <a href="https://github.com/HorizonRobotics/BIP3D" target="_blank" style="margin: 2px;">
    <img alt="Code" src="https://img.shields.io/badge/Code-Github-bule" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://linxuewu.github.io/BIP3D-page/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/Homepage-BIP3D-green" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/HorizonRobotics/BIP3D" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/Models-Hugging%20Face-yellow" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2411.14869" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-Arxiv-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


## :rocket: News
**14/Mar/2025**: Our code has been released.

**27/Feb/2025**: Our paper has been accepted by CVPR 2025.

**22/Nov/2024**: We release our paper to [Arxiv](https://arxiv.org/abs/2411.14869).

## :open_book: Quick Start
[Quick Start](docs/quick_start.md)

## :link: Framework
<div align="center">
  <img src="https://github.com/HorizonRobotics/BIP3D/raw/main/resources/bip3d_structure.png" width="90%" alt="BIP3D" />
  <p style="font-size:0.8em; color:#555;">The Architecture Diagram of BIP3D, where the red stars indicate the parts that have been modified or added compared to the base model, GroundingDINO, and dashed lines indicate optional elements.</p>
</div>

## :trophy: Results on EmbodiedScan Benchmark
We made several improvements based on the original paper, achieving better 3D perception results. The main improvements include the following two points:
1. **New Fusion Operation**: We enhanced the decoder by replacing the deformable aggregation (DAG) with a 3D deformable attention mechanism (DAT). Specifically, we improved the feature sampling process by transitioning from bilinear interpolation to trilinear interpolation, which leverages depth distribution for more accurate feature extraction.
2. **Mixed Data Training**: To optimize the grounding model's performance, we adopted a mixed-data training strategy by integrating detection data with grounding data during the grounding finetuning process.

### 1. Results on Multi-view 3D Detection Validation Dataset
Op DAG denotes deformable aggregation, and DAT denotes 3D deformable attention. Set [`with_depth=True`](configs/bip3d_det.py#L175) to activate the DAT.

The metric in the table is `AP@0.25`. For more metrics, please refer to the logs.
|Model | Inputs | Op | Overall | Head | Common | Tail | Small | Medium | Large | ScanNet | 3RScan | MP3D | ckpt | log |
|  :----  | :---: |  :---: | :---: |:---: | :---: | :---: | :---:| :---:|:---:|:---: | :---: | :----: | :----: | :---: |
|[BIP3D](configs/bip3d_det_rgb.py) | RGB | DAG | 16.57|23.29|13.84|12.29|2.67|17.85|12.89|19.71|26.76|8.50   | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/det_rgb_dag/model_checkpoint.pth) | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/det_rgb_dag/job.log) |
|[BIP3D](configs/bip3d_det_rgb.py) | RGB | DAT | 16.67|22.41|14.19|13.18|3.32|17.25|14.89|20.80|24.18|9.91  | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/det_rgb_dat/model_checkpoint.pth) | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/det_rgb_dat/job.log) |
|[BIP3D](configs/bip3d_det.py) |RGB-D | DAG | 22.53|28.89|20.51|17.83|6.95|24.21|15.46|24.77|35.29|10.34  | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/det_rgbd_dag/model_checkpoint.pth) | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/det_rgbd_dag/job.log) |
|[BIP3D](configs/bip3d_det.py) |RGB-D | DAT | 23.24|31.51|20.20|17.62|7.31|24.09|15.82|26.35|36.29|11.44   | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/det_rgbd_dat/model_checkpoint.pth) | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/det_rgbd_dat/job.log) |

### 2. Results on Multi-view 3D Grounding Mini Dataset
To train and validate with mini dataset, set [`data_version="v1-mini"`](configs/bip3d_grounding.py#L333).
|Model | Inputs | Op | Overall | Easy | Hard | View-dep | View-indep | ScanNet | 3RScan | MP3D | ckpt | log |
|  :----  | :---: | :---: | :---: | :---: | :---:| :---:|:---:|:---: | :---: | :----: |:---: | :----: |
|[BIP3D](configs/bip3d_grounding_rgb.py) | RGB | DAG | 44.00|44.39|39.56|46.05|42.92|48.62|42.47|36.40  | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_mini_rgb_dag/model_checkpoint.pth) | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_mini_rgb_dag/job.log) |
|[BIP3D](configs/bip3d_grounding_rgb.py) | RGB | DAT | 44.43|44.74|41.02|45.17|44.04|49.70|41.81|37.28  | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_mini_rgb_dat/model_checkpoint.pth) | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_mini_rgb_dat/job.log) |
|[BIP3D](configs/bip3d_grounding.py) | RGB-D | DAG | 45.79|46.22|40.91|45.93|45.71|48.94|46.61|37.36  | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_mini_rgbd_dag/model_checkpoint.pth) | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_mini_rgbd_dag/job.log) |
|[BIP3D](configs/bip3d_grounding.py) | RGB-D | DAT | 58.47|59.02|52.23|60.20|57.56|66.63|54.79|46.72  | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_mini_rgbd_dat/model_checkpoint.pth) | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_mini_rgbd_dat/job.log) |


### 3. Results on Multi-view 3D Grounding Validation Dataset
|Model | Inputs | Op | Mixed Data | Overall | Easy | Hard | View-dep | View-indep | ScanNet | 3RScan | MP3D | ckpt | log |
|  :----  | :---: | :---: | :---: |:---: | :---: | :---:| :---:|:---:|:---: | :---: | :----: |:---: | :----: |
|[BIP3D](configs/bip3d_grounding_rgb.py) | RGB | DAG |No| 45.81|46.21|41.34|47.07|45.09|50.40|47.53|32.97   | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_rgb_dag/model_checkpoint.pth) | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_rgb_dag/job.log) |
|[BIP3D](configs/bip3d_grounding_rgb.py) | RGB | DAT |No| 47.29|47.82|41.42|48.58|46.56|52.74|47.85|34.60   | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_rgb_dat/model_checkpoint.pth) | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_rgb_dat/job.log) |
|[BIP3D](configs/bip3d_grounding.py) | RGB-D | DAG |No| 53.75|53.87|52.43|55.21|52.93|60.05|54.92|38.20   | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_rgbd_dag/model_checkpoint.pth) | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_rgbd_dag/job.log) |
|[BIP3D](configs/bip3d_grounding.py) | RGB-D | DAT |No|61.36|61.88|55.58|62.43|60.76|66.96|62.75|46.92   | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_rgbd_dat/model_checkpoint.pth) | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_rgbd_dat/job.log) |
|[BIP3D](configs/bip3d_det_grounding.py) | RGB-D | DAT |Yes|66.58|66.99|62.07|67.95|65.81|72.43|68.26|51.14   | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_rgbd_dat_mixdata/model_checkpoint.pth) | [link](https://huggingface.co/HorizonRobotics/BIP3D/blob/main/grounding_rgbd_dat_mixdata/job.log) |


### 4. [Results on Multi-view 3D Grounding Test Dataset](https://huggingface.co/spaces/AGC2024/visual-grounding-2024)

|Model | Overall | Easy | Hard | View-dep | View-indep | ckpt | log |
|  :----  | :---: | :---: | :---: | :---: | :---:| :---:|:---:|
|[EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan) | 39.67 | 40.52 | 30.24 | 39.05 | 39.94 | - | - |
|[SAG3D*](https://opendrivelab.github.io/Challenge%202024/multiview_Mi-Robot.pdf) | 46.92 | 47.72 | 38.03 | 46.31 | 47.18 | - | - |
|[DenseG*](https://opendrivelab.github.io/Challenge%202024/multiview_THU-LenovoAI.pdf) | 59.59 | 60.39 | 50.81 | 60.50 | 59.20 |  - | - |
|[BIP3D](configs/bip3d_det_grounding.py) | 67.38 | 68.12 | 59.08 | 67.88 | 67.16 |  - | - |
|BIP3D-B | 70.53 | 71.22 | 62.91 | 70.69 | 70.47 | - | - |

`*` denotes model ensemble, and note that our BIP3D does not use the ensemble trick. This differs from what is mentioned in the paper and shows significant improvements.

Our best model, BIP3D-B, is based on GroundingDINO-base and is trained with the addition ARKitScenes dataset.


## :page_facing_up: Citation
```
@article{lin2024bip3d,
  title={BIP3D: Bridging 2D Images and 3D Perception for Embodied Intelligence},
  author={Lin, Xuewu and Lin, Tianwei and Huang, Lichao and Xie, Hongyu and Su, Zhizhong},
  journal={arXiv preprint arXiv:2411.14869},
  year={2024}
}
```


## :handshake: Acknowledgement
[EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan)

[Sparse4D](https://github.com/HorizonRobotics/Sparse4D)

[3D-deformable-attention](https://github.com/IDEA-Research/3D-deformable-attention)

[mmdet-GroundingDINO](https://github.com/open-mmlab/mmdetection/tree/main/configs/grounding_dino)
