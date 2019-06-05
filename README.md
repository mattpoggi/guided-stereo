# Guided Stereo Matching

Demo code of "Guided Stereo Matching", [Matteo Poggi](https://vision.disi.unibo.it/~mpoggi/), Davide Pallotti, [Fabio Tosi](https://vision.disi.unibo.it/~ftosi/) and [Stefano Mattoccia](https://vision.disi.unibo.it/~smatt/Site/Home.html), CVPR 2019. 

### License
Copyright (c) 2019 University of Bologna. Patent pending. All rights reserved. Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode ).

**NOTE: This code is for demonstration purposes. We do not plan to release training code.**

![Alt text](./images/guided.png?raw=true "Guided Stereo Matching")

[[Paper]](https://vision.disi.unibo.it/~mpoggi/papers/cvpr2019guided.pdf) - [[Poster]](https://vision.deis.unibo.it/~mpoggi/papers/cvpr2019guided_poster.pdf) - [[Youtube Video]](https://www.youtube.com/watch?v=AVlPu3K2ays)

## Citation
```shell
@inproceedings{Poggi_CVPR_2019,
  title     = {Guided Stereo Matching},
  author    = {Poggi, Matteo and
               Pallotti, Davide and
               Tosi, Fabio and
               Mattoccia, Stefano},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```   

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)
5. [Acknowledgements](#acknowledgements)

## Introduction

Stereo is a prominent technique to infer dense depth maps from images, and deep learning further pushed forward the state-of-the-art, making end-to-end architectures unrivaled when enough data is available for training. However, deep networks suffer from significant drops in accuracy  when dealing with new environments. Therefore, in this paper, we introduce Guided Stereo Matching, a novel paradigm leveraging a small amount of sparse, yet reliable depth measurements retrieved from an external source enabling to ameliorate this weakness. The additional sparse cues required by our method can be obtained with any strategy (e.g., a LiDAR) and used to enhance features linked to corresponding disparity hypotheses. Our formulation is general and fully differentiable, thus enabling to exploit the additional sparse inputs in pre-trained deep stereo networks as well as for training a new instance from scratch. Extensive experiments on three standard datasets and two stateof-the-art
deep architectures show that even with a small set of sparse input cues, i) the proposed paradigm enables significant improvements to pre-trained networks. Moreover, ii) training from scratch notably increases accuracy and robustness to domain shifts. Finally, iii) it is suited and effective even with traditional stereo algorithms such as SGM.

## Usage

### Requirements

* `PyTorch 0.4` (recommended) 
* `python packages` such as opencv, PIL, numpy

### Getting started

Download KITTI demo sequence and pretrained models running

```shell
sh get_weights_and_data.sh
```

### Run the demo

Launch the following command

```shell
python run.py --datapath [sequence_path] \ 
              --loadmodel [model_path] \
              --output_dir [output_path] \
              --guided \
              --display \
              --save \
              --verbose \
```
Optional arguments:
* `--guided`: enables guided stereo
* `--display`: shows results on screen 
* `--save`: saves results in `output_dir` 
* `--verbose`: prints single stereo pair stats 

## Results

Results on the provided sequence `2011_09_26_0011`:

|  Model           | bad2-All (%) | bad2-Nog (%) | MAE-All | MAE-Nog | Density (%) |
|------------------|--------------|--------------|---------|---------|-------------|
|  PSMnet-ft       |     1.71     |     1.73     |   0.72  |   0.72  |       -     |
|  PSMnet-ft-gd    |     1.13     |     1.15     |   0.60  |   0.61  |     3.68    |
|  PSMnet-ft-gd-tr |     0.67     |     0.67     |   0.47  |   0.47  |     3.68    |

### Qualitative results
Qualitative results on Middlebury v3 sampling 5% hints from ground truth. From left to right, reference input image (a), disparity map by PSMNet (b), PSMNet-gd-tr (c) and ground truth (d).

![Alt text](./images/qualitative.png?raw=true "Qualitative results on Middlebury v3")

## Contacts
m [dot] poggi [at] unibo [dot] it

## Acknowledgements

Thanks to Jia-Ren Chang for sharing the original implementation of PSMNet: https://github.com/JiaRenChang/PSMNet
