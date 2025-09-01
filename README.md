# PX OmniSharing Toolkit

[TOC]

PX OmniSharing is a comprehensive toolkit for data processing and sharing. It is the freeware version of our data post-processing tool, supporting a complete workflow from raw human data to our model’s training data. 

The toolkit has three parts: **PX Hand Pose**, **PX Retargeting**, and **PX Replay**. They are in the directories, [px_hand_pose](px_hand_pose), [px_retargeting](px_retargeting), and [px_replay](px_replay).   

# Overview 

![PaXini EID Workflow](images/data_flow.png)

Our processing pipeline involves two data categories: Phase 1 and Phase 2.

| Category | Description |
|----------|----------|
| **Phase 1**   | Raw data after preprocessing. HDF5 file. |
| **Phase 2**   | Data retargeted to the DH13 configuration; <br>can be organized for VTLA model training. HDF5 file. |

The pipeline is separated into 2 stages: **PX Hand Pose** and **PX Retargeting**             
| | Input | Output |
|------|----------|----------|
| **PX Hand Pose**   | Phase 1 | 6D Bracelet Poses |
| **PX Retargeting**   | a. Phase 1 <br> b. 6D Bracelet Poses  | Phase 2 |

Video and audio recorded during data acquisition are included in the final Phase 2 output.    

**PX Replay** is the visualization module.

# Project Structure
```text
.
├── images
│ 
├── px_hand_pose
│   ├── experimental
│   ├── FoundationPose
│   ├── iinet
│   ├── main # scripts of px_hand_pose
│   ├── README.md
│   └── utils
│ 
├── px_replay
│   ├── asset
│   ├── env.sh
│   ├── pyproject.toml
│   ├── README.md
│   ├── requirements.txt
│   ├── src
│   └── visual_scripts # scripts of px_replay
│ 
├── px_retargeting
│   ├── config
│   ├── env.sh
│   ├── install.sh
│   ├── px_retargeting-1.1.0-cp310-cp310-linux_x86_64.whl
│   ├── README.md
│   ├── third_party
│   └── urdfs
│ 
└── README.md
```

# Data Structure
## Phase 1 data
```
*.hdf5
└── episode_[INDEX]_[HHMMSS]_[ROOM_ID]_[PERSONNEL].hdf5
    ├── dataset 
    │   └── observation                 
    │       ├── audio                            # compressed audio, including task description    
    │       ├── image                            # image from data acquisition
    │       │   ├── attributes           
    │       │   ├── rgbd_rgb_extrinsic
    │       │   ├── RGB_CameraXXX                # RGB camera
    │       │   │   ├── data                     # shape: (x, ), compressed video file
    │       │   │   ├── extrinsics       
    │       │   │   └── intrinsics       
    │       │   ├── RGBD_CameraXXX               # RGBD camera
    │       │   │   ├── color
    │       │   │   │   ├── data                 # shape: (x, ), compressed video file      
    │       │   │   │   └── intrinsics 
    │       │   │   ├── extrinsics 
    │       │   │   ├── inner_extrinsics 
    │       │   │   ├── left
    │       │   │   │   ├── data                 # shape: (x, ), compressed video file      
    │       │   │   │   └── intrinsics 
    │       │   │   ├── right
    │       │   │   │   ├── data                 # shape: (x, ), compressed video file      
    │       │   │   │   └── intrinsics 
    │       │   │   └── timestamp
    │       │   └── [...]                        # other cameras   
    │       │
    │       └────── state
    │               ├── lefthand                 
    │               │   ├── encoder              
    │               │   │   ├── data             # shape: (n, 29)
    │               │   │   ├── timestamp
    │               │   │   └── attributes       
    │               │   └── tactile              
    │               │       ├── data             # shape: (x, )
    │               │       ├── timestamp
    │               │       └── attributes       
    │               └── righthand                 
    │                   ├── encoder              
    │                   │   ├── data             # shape: (n, 29)
    │                   │   ├── timestamp
    │                   │   └── attributes       
    │                   └── tactile              
    │                       ├── data             # shape: (x, )
    │                       ├── timestamp
    │                       └── attributes   
    └── info
        ├── errors
        └── attributes
```

## Phase 2 data
```
*_dh13.hdf5
└── dataset     
    └── observation                  
        ├── audio                    # compressed audio, including task description
        ├── image                    # image from data acquisition
        │   ├── attributes           
        │   ├── rgbd_rgb_extrinsic
        │   ├── RGB_CameraXXX        # RGB camera
        │   │   ├── data             # shape: (x,), compressed video file
        │   │   ├── extrinsics       
        │   │   └── intrinsics       
        │   ├── RGBD_CameraXXX       # RGBD camera
        │   │   ├── data             # shape: (x,), compressed video file
        │   │   ├── extrinsics       
        │   │   └── intrinsics       
        │   └── [...]                # other cameras
        │
        ├── lefthand                 
        │   ├── attributes           # dexh13_hand_left
        │   ├── joints               
        │   │   ├── data             # shape: (n, 16)
        │   │   └── attributes       # joint_names: 'left_index_joint_0' 'left_index_joint_1' ...
        │   ├── handpose             
        │   │   ├── data             # shape: (n, 7)
        │   │   └── attributes       # order: [x, y, z, qw, qx, qy, qz]; Cartesian coordinates followed by quaternions 
        │   └── tactile              
        │       ├── data             # shape: (n, 2580)
        │       └── attributes       # sensor_lengths and sensor_names
        └── righthand                
            ├── attributes           # dexh13_hand_right
            ├── joints               
            │   ├── data             # shape: (n, 16)
            │   └── attributes       # joint_names: 'right_index_joint_0' 'right_index_joint_1'
            ├── handpose             
            │   ├── data             # shape: (n, 7)
            │   └── attributes       # order: [x, y, z, qw, qx, qy, qz]; Cartesian coordinates followed by quaternions 
            └── tactile             
                ├── data             # shape: (n, 2580)
                └── attributes       # sensor_lengths and sensor_names
```
# License

| Component | Type | License | Commercial Use |
|-----------|------|---------|----------------|
| Python Package (.whl) | Binary | Proprietary | Requires License |
| Python Package (.py) | Source | MIT | Yes |
| Documentation | Source | MIT | Yes |
| Bracelet Detection Model | Model weights (.pt) | Proprietary | Requires License |
| Datasets | Source | CC-BY-NC-SA 4.0 | No |

## Installation & Usage Rights

Research/Education Use

# Acknowledgment
We would like to thank the authors of [IINet](https://github.com/blindwatch/IINet), [FoundationStereo](https://github.com/NVlabs/FoundationStereo), [FoundationPose](https://github.com/NVlabs/FoundationPose), [YOLO](https://github.com/ultralytics/ultralytics), [manotorch](https://github.com/lixiny/manotorch), and [PyTorch Robot Kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) for releasing their code and/or model, which we built upon in our open toolkit. Also, thanks to the authors of [FoundationPose++](https://github.com/teal024/FoundationPose-plus-plus) for their very helpful Kalman filter.
