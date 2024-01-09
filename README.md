# SiT-MLP

This is the official implementation of our paper [SiT-MLP: A Simple MLP with Point-wise Topology Feature Learning for Skeleton-based Action Recognition](https://arxiv.org/abs/2308.16018)

## Abstract

Graph convolution networks (GCNs) have achieved remarkable performance in skeleton-based action recognition. However, previous GCN-based methods rely on elaborate human priors excessively and construct complex feature aggregation mechanisms, which limits the generalizability and effectiveness of networks. To solve these problems, we propose a novel Spatial Topology Gating Unit (STGU), an MLP-based variant without extra priors, to capture the co-occurrence topology features that encode the spatial dependency across all joints. In STGU, to learn the point-wise topology features, a new gate-based feature interaction mechanism is introduced to activate the features point-to-point by the attention map generated from the input sample. Based on the STGU, we propose the first MLP-based model, SiTMLP, for skeleton-based action recognition in this work. Compared with previous methods on three large-scale datasets, SiTMLP achieves competitive performance. In addition, SiT-MLP reduces the parameters by up to 62.5% with favorable results.

## Experiment

|                      Model                       | Parameters | FLOPs | Accuracy |
| :----------------------------------------------: | :--------: | :---: | :------: |
|  [2S-AGCN](https://github.com/lshiwjx/2s-AGCN)   |    3.5M    | 3.9G  |   95.1   |
| [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) |    1.4M    | 1.8G  |   96.8   |
|  [InfoGCN](https://github.com/stnoah1/infogcn)   |    1.5M    | 1.7G  |   96.7   |
|                     SiT-MLP                      |    0.6M    | 0.7G  |   96.8   |

![result](https://github.com/BUPTSJZhang/Ta-MLP/blob/main/resource/result.jpg)

Comparsion of performance and parameter size on X-sub benchmark of NTU RGB+D 60 dataset. We report the accuracy as performance on the vertical axis. The closer to the top-left, the better.

## Ta-MLP architecture

![architecture](https://github.com/BUPTSJZhang/Ta-MLP/blob/main/resource/architecture.jpg)

![framewrok](https://github.com/BUPTSJZhang/Ta-MLP/blob/main/resource/framework.jpg)

![fc](https://github.com/BUPTSJZhang/Ta-MLP/blob/main/resource/fc.jpg)

## Preparation
- [ ] We will release our code soon!
