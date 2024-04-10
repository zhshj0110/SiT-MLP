# SiT-MLP

This is the official implementation of our paper [SiT-MLP: A Simple MLP with Point-wise Topology Feature Learning for Skeleton-based Action Recognition](https://arxiv.org/abs/2308.16018)

Note: Our approch is MLP-based and GCN-free. The graph folder is adopted for different modality.

## Abstract
Graph convolution networks (GCNs) have achieved remarkable performance in skeleton-based action recognition. However, previous GCN-based methods rely on elaborate human priors excessively and construct complex feature aggregation mechanisms, which limits the generalizability and effectiveness of networks. To solve these problems, we propose a novel Spatial Topology Gating Unit (STGU), an MLP-based variant without extra priors, to capture the co-occurrence topology features that encode the spatial dependency across all joints. In STGU, to learn the point-wise topology features, a new gate-based feature interaction mechanism is introduced to activate the features point-to-point by the attention map generated from the input sample. Based on the STGU, we propose the first MLP-based model, SiT-MLP, for skeleton-based action recognition in this work. Compared with previous methods on three large-scale datasets, SiT-MLP achieves competitive performance. In addition, SiT-MLP reduces the parameters significantly with favorable results.

## Experiment
|                      Model                       | Parameters | FLOPs | Accuracy |
| :----------------------------------------------: | :--------: | :---: | :------: |
|  [2S-AGCN](https://github.com/lshiwjx/2s-AGCN)   |    3.5M    | 3.9G  |   95.1   |
| [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) |    1.4M    | 1.8G  |   96.8   |
|  [InfoGCN](https://github.com/stnoah1/infogcn)   |    1.5M    | 1.7G  |   96.7   |
|                     SiT-MLP                      |    0.6M    | 0.7G  |   96.8   |

![result](https://github.com/BUPTSJZhang/Ta-MLP/blob/main/resource/result.png)

Comparsion of performance and parameter size on X-sub benchmark of NTU RGB+D 60 dataset. We report the accuracy as performance on the vertical axis. The closer to the top-left, the better.

## SiT-MLP architecture

<!-- ![architecture](https://github.com/BUPTSJZhang/Ta-MLP/blob/main/resource/architecture.png) -->

![framewrok](https://github.com/BUPTSJZhang/Ta-MLP/blob/main/resource/framework.png)

<!-- ![fc](https://github.com/BUPTSJZhang/Ta-MLP/blob/main/resource/fc.png) -->



# Prerequisites
You can install all dependencies by running ```pip install -r requirements.txt```  <br />
Then, you need to install torchlight by running ```pip install -e torchlight```  <br />

# Data Preparation
### Download datasets.

#### There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

#### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

#### NW-UCLA

1. Download dataset from [here](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0)
2. Move `all_sqe` to `./data/NW-UCLA`

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

# Training & Testing

### Training
- To train model on NTU60/120

```
# Example: training SiT-MLP on NTU RGB+D cross subject joint modality
CUDA_VISIBLE_DEVICES=0,1 python main.py --config config/nturgbd-cross-subject/mlp_joint.yaml 
# Example: training SiT-MLP on NTU RGB+D cross subject bone modality
CUDA_VISIBLE_DEVICES=0,1 python main.py --config config/nturgbd-cross-subject/mlp_bone.yaml 
# Example: training SiT-MLP on NTU RGB+D 120 cross subject joint modality
CUDA_VISIBLE_DEVICES=0,1 python main.py --config config/nturgbd120-cross-subject/mlp_joint.yaml 
# Example: training SiT-MLP on NTU RGB+D 120 cross subject bone modality
CUDA_VISIBLE_DEVICES=0,1 python main.py --config config/nturgbd120-cross-subject/mlp_bone.yaml 
```


- To train model on NW-UCLA

```
CUDA_VISIBLE_DEVICES=0,1 python main.py --config config/ucla/mlp_joint.yaml 
```


### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt
```

- To ensemble the results of different modalities, run 
```
# Example: ensemble four modalities of SiT-MLP on NTU RGB+D 120 cross subject
python ensemble.py --datasets ntu120/xsub --position_ckpts work_dir/ntu120/xsub/mlp/joint work_dir/ntu120/xsub/mlp/bone --motion_ckpts work_dir/ntu120/xsub/mlp/joint_vel work_dir/ntu120/xsub/mlp/bone_vel
```

## Acknowledgements
This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch). The code for different modality is adopted from [InfoGCN](https://github.com/stnoah1/infogcn).

# Citation

Please cite this work if you find it useful:
```BibTex
@article{zhang2023sitmlp,
    author={Zhang, Shaojie and Yin, Jianqin and Dang, Yonghao and Fu, Jiajun},
    journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
    title={SiT-MLP: A Simple MLP with Point-wise Topology Feature Learning for Skeleton-based Action Recognition}, 
    year={2024},
    doi={10.1109/TCSVT.2024.3386553}
}
```
