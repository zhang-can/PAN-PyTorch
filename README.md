# PAN: Persistent Appearance Network

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pan-towards-fast-action-recognition-via/action-recognition-in-videos-on-something-1)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something-1?p=pan-towards-fast-action-recognition-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pan-towards-fast-action-recognition-via/action-recognition-in-videos-on-something)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something?p=pan-towards-fast-action-recognition-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pan-towards-fast-action-recognition-via/action-recognition-in-videos-on-jester)](https://paperswithcode.com/sota/action-recognition-in-videos-on-jester?p=pan-towards-fast-action-recognition-via)

PyTorch Implementation of paper:

> **PAN: Towards Fast Action Recognition via Learning Persistence of Appearance**
>
> Can Zhang, Yuexian Zou\*, Guang Chen and Lei Gan.
>
> [[ArXiv](https://arxiv.org/abs/2008.03462)]

## Updates

**[12 Aug 2020]** We have released the codebase and models of the PAN. 

## Main Contribution

Efficiently modeling dynamic motion information in videos is crucial for action recognition task. Most state-of-the-art methods heavily rely on dense optical flow as motion representation. Although combining optical flow with RGB frames as input can achieve excellent recognition performance, the optical flow extraction is very time-consuming. This undoubtably will count against real-time action recognition. In this paper, we shed light on **fast action recognition** by lifting the reliance on optical flow. We design a novel **motion cue** called **Persistence of Appearance (PA)** that focuses more on distilling the motion information at boundaries. Extensive experiments show that our PA is over 1000x faster (8196fps *vs.* 8fps) than conventional optical flow in terms of motion modeling speed. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/32992487/89706349-56563200-d997-11ea-8ed6-4ceca2883bad.gif" />
</p>

## Content

- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Core Codes](#core-codes)
  - [PA Module](#pa-module)
  - [VAP Module](#vap-module)
- [Pretrained Models](#pretrained-models)
  + [Something-Something-V1](#something-something-v1)
  + [Something-Something-V2](#something-something-v2)
- [Testing](#testing)
- [Training](#training)
- [Other Info](#other-info)
  - [References](#references)
  - [Citation](#citation)
  - [Contact](#contact)

## Dependencies

Please make sure the following libraries are installed successfully:

- [PyTorch](https://pytorch.org/) >= 1.0
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm.git)
- [scikit-learn](https://scikit-learn.org/stable/)

## Data Preparation

Following the common practice, we need to first extract videos into frames for fast reading. Please refer to [TSN](https://github.com/yjxiong/temporal-segment-networks) repo for the detailed guide of data pre-processing. We have successfully trained on [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/), [UCF101](http://crcv.ucf.edu/data/UCF101.php), [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/), [Something-Something-V1](https://20bn.com/datasets/something-something/v1) and [V2](https://20bn.com/datasets/something-something/v2), [Jester](https://20bn.com/datasets/jester) datasets with this codebase. Basically, the processing of video data can be summarized into 3 steps:

1. Extract frames from videos:

   * For Something-Something-V2 dataset, please use [tools/vid2img_sthv2.py](tools/vid2img_sthv2.py) 

   * For Kinetics dataset, please use [tools/vid2img_kinetics.py](tools/vid2img_kinetics.py) 

2. Generate file lists needed for dataloader:

   * Each line of the list file will contain a tuple of (*extracted video frame folder name, video frame number, and video groundtruth class*). A list file looks like this:

     ```
     video_frame_folder 100 10
     video_2_frame_folder 150 31
     ...
     ```

   * Or you can use off-the-shelf tools provided by other repos:
     * For Something-Something-V1 & V2 datasets, please use [tools/gen_label_sthv1.py](tools/gen_label_sthv1.py) & [tools/gen_label_sthv2.py](tools/gen_label_sthv2.py)
     * For Kinetics dataset, please use [tools/gen_label_kinetics.py](tools/gen_label_kinetics.py)

3. Add the information to [ops/dataset_configs.py](ops/dataset_configs.py)

## Core Codes

### PA Module

<p align="center">
  <img src="https://images.gitee.com/uploads/images/2020/0812/102952_5cf9a077_1652972.png" width="400px" />
</p>

PA module aims to speed up the motion modeling procedure, it can be simply injected at the bottom of the network to lift the reliance on optical flow.

```python
from ops.PAN_modules import PA

PA_module = PA(n_length=4) # adjacent '4' frames are sampled for computing PA
# shape of x: [N*T*m, 3, H, W]
x = torch.randn(5*8*4, 3, 224, 224)
# shape of PA_out: [N*T, m-1, H, W]
PA_out = PA_module(x) # torch.Size([40, 3, 224, 224])
```

### VAP Module

VAP module aims to adaptively emphasize expressive features and suppress less informative ones by observing global information across various timescales. It is adopted at the top of the network to achieve long-term temporal modeling.

<p align="center">
  <img src="https://images.gitee.com/uploads/images/2020/0812/102950_06ec8516_1652972.png" width="300px" />
</p>

```python
from ops.PAN_modules import VAP

VAP_module = VAP(n_segment=8, feature_dim=2048, num_class=174, dropout_ratio=0.5)
# shape of x: [N*T, D]
x = torch.randn(5*8, 2048)
# shape of VAP_out: [N, num_class]
VAP_out = VAP_module(x) # torch.Size([5, 174])
```

## Pretrained Models

Here, we provide the pretrained models of PAN models on Something-Something-V1 & V2 datasets. Recognizing actions in these datasets requires strong temporal modeling ability, as many action classes are symmetrical. PAN achieves state-of-the-art performance on these datasets. Notably, our method even surpasses optical flow based methods while with only RGB frames as input.

### Something-Something-V1

<div align="center">
<table>
<thead>
<tr>
<th align="center">Model</th>
<th align="center">Backbone</th>
<th align="center">FLOPs * views</th>
<th align="center">Val Top1</th>
<th align="center">Val Top5</th>
<th align="center">Checkpoints</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">PAN<sub>Lite</sub></td>
<td align="center" rowspan="3">ResNet-50</td>
<td align="center">35.7G * 1</td>
<td align="center">48.0</td>
<td align="center">76.1</td>
<td align="center" rowspan="3">[<a href="https://drive.google.com/drive/folders/1Be3o8pesPrk8uEoBNd4OW6qtKkeXoVBU?usp=sharing">Google Drive</a>] or [<a href="https://share.weiyun.com/F2PJnUiE" rel="nofollow">Weiyun</a>]</td>
</tr>
<tr>
<td align="center">PAN<sub>Full</sub></td>
<td align="center">67.7G * 1</td>
<td align="center">50.5</td>
<td align="center">79.2</td>
</tr>
<tr>
<td align="center">PAN<sub>En</sub></td>
<td align="center">(46.6G+88.4G) * 2</td>
<td align="center">53.4</td>
<td align="center">81.1</td>
</tr>
<tr>
<td align="center">PAN<sub>En</sub></td>
<td align="center">ResNet-101</td>
<td align="center">(85.6G+166.1G) * 2</td>
<td align="center">55.3</td>
<td align="center">82.8</td>
<td align="center">[<a href="https://drive.google.com/drive/folders/1sD242c3rkhIjTPxJiyPnXRx7zKLnA12b?usp=sharing">Google Drive</a>] or [<a href="https://share.weiyun.com/bqouigPA" rel="nofollow">Weiyun</a>]</td>
</tr>
</tbody>
</table>
</div>

### Something-Something-V2

<div align="center">
<table>
<thead>
<tr>
<th align="center">Model</th>
<th align="center">Backbone</th>
<th align="center">FLOPs * views</th>
<th align="center">Val Top1</th>
<th align="center">Val Top5</th>
<th align="center">Checkpoints</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">PAN<sub>Lite</sub></td>
<td align="center" rowspan="3">ResNet-50</td>
<td align="center">35.7G * 1</td>
<td align="center">60.8</td>
<td align="center">86.7</td>
<td align="center" rowspan="3">[<a href="https://drive.google.com/drive/folders/1dqMphONnGLBxhArMazr281Ix4qOX58MI?usp=sharing">Google Drive</a>] or [<a href="https://share.weiyun.com/oXQY3rx7" rel="nofollow">Weiyun</a>]</td>
</tr>
<tr>
<td align="center">PAN<sub>Full</sub></td>
<td align="center">67.7G * 1</td>
<td align="center">63.8</td>
<td align="center">88.6</td>
</tr>
<tr>
<td align="center">PAN<sub>En</sub></td>
<td align="center">(46.6G+88.4G) * 2</td>
<td align="center">66.2</td>
<td align="center">90.1</td>
</tr>
<tr>
<td align="center">PAN<sub>En</sub></td>
<td align="center">ResNet-101</td>
<td align="center">(85.6G+166.1G) * 2</td>
<td align="center">66.5</td>
<td align="center">90.6</td>
<td align="center">[<a href="https://drive.google.com/drive/folders/1NL-CNQPLfV3abnJDMcz3AF4WFsH7LDhN?usp=sharing">Google Drive</a>] or [<a href="https://share.weiyun.com/oSOgog0J" rel="nofollow">Weiyun</a>]</td>
</tr>
</tbody>
</table>
</div>

## Testing 

For example, to test the PAN models on Something-Something-V1, you can first put the downloaded `.pth.tar` files into the "pretrained" folder and then run:

```bash
# test PAN_Lite
bash scripts/test/sthv1/Lite.sh

# test PAN_Full
bash scripts/test/sthv1/Full.sh

# test PAN_En
bash scripts/test/sthv1/En.sh
```

## Training 

We provided several scripts to train PAN with this repo, please refer to "[scripts](scripts/)" folder for more details. For example, to train PAN on Something-Something-V1, you can run:

```bash
# train PAN_Lite
bash scripts/train/sthv1/Lite.sh

# train PAN_Full RGB branch
bash scripts/train/sthv1/Full_RGB.sh

# train PAN_Full PA branch
bash scripts/train/sthv1/Full_PA.sh
```

Notice that you should scale up the learning rate with batch size. For example, if you use a batch size of 256 you should set learning rate to 0.04.

## Other Info

### References

This repository is built upon the following baseline implementations for the action recognition task.

- [TSM](https://github.com/mit-han-lab/temporal-shift-module)
- [TSN](https://github.com/yjxiong/tsn-pytorch)

### Citation

Please **[★star]** this repo and **[cite]** the following arXiv paper if you feel our PAN useful to your research:

```
@misc{zhang2020pan,
    title={PAN: Towards Fast Action Recognition via Learning Persistence of Appearance},
    author={Can Zhang and Yuexian Zou and Guang Chen and Lei Gan},
    year={2020},
    eprint={2008.03462},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

Or if you prefer "publication", you can cite our preliminary work on ACM MM 2019:

```
@inproceedings{zhang2019pan,
  title={PAN: Persistent Appearance Network with an Efficient Motion Cue for Fast Action Recognition},
  author={Zhang, Can and Zou, Yuexian and Chen, Guang and Gan, Lei},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={500--509},
  year={2019}
}
```

### Contact

For any questions, please feel free to open an issue or contact:

```
Can Zhang: zhang.can.pku@gmail.com
```
