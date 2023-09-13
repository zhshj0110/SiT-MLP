# Ta-MLP

This is the official implementation of our paper [Topology-aware MLP for Skeleton-based Action
Recognition](https://arxiv.org/abs/2308.16018)



## Experiment

|                      Model                       | Parameters | FLOPs | Accuracy |
| :----------------------------------------------: | :--------: | :---: | :------: |
|  [2S-AGCN](https://github.com/lshiwjx/2s-AGCN)   |    3.5M    | 3.9G  |   95.1   |
| [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) |    1.4M    | 1.8G  |   96.8   |
|  [InfoGCN](https://github.com/stnoah1/infogcn)   |    1.5M    | 1.7G  |   96.7   |
|                      Ta-MLP                      |    0.6M    | 0.7G  |   96.8   |

![result](https://github.com/BUPTSJZhang/Ta-MLP/blob/main/resource/result.jpg)

Comparsion of performance and parameter size on X-sub benchmark of NTU RGB+D 60 dataset. We report the accuracy as performance on the vertical axis. The closer to the top-left, the better.

## Ta-MLP architecture

![architecture](https://github.com/BUPTSJZhang/Ta-MLP/blob/main/resource/architecture.jpg)

![framewrok](https://github.com/BUPTSJZhang/Ta-MLP/blob/main/resource/framework.jpg)

![fc](https://github.com/BUPTSJZhang/Ta-MLP/blob/main/resource/fc.jpg)

## Preparation
- [ ] We will release our code soon!
