# Data Structure in the Pipeline

# Overview
This document defines the directory layout and HDF5 group structure for each pipeline stage. 

| Stage | File | Description |
|----------|----------| -----------------|
| **DF-1** | HDF5 | The overall input: raw data after preprocessing and quality inspection |
| **DF-2** | HDF5 | 1st output: DF-1 with encoder and tactile data parsed; adds bimanual and object poses; contains both **action** and **observation** |
| **DF-2R** | HDF5 |  2nd output: DF-2 retargeted to a dexterous hand model |
| **DF-3**  | LeRobot Dataset | 3rd output: converts DF-2R to the LeRobot dataset format; can be used for VLA model training |

# Content

## DF-1: Raw Data
### File layout
```text
batch_XXX/
└── episode_{INDEX}_{HHMMSS}_{ROOM-ID}_{PERSONNEL-ID}.hdf5
```

### HDF5 structure
```text
/dataset
├── meta                          # Task description
└── observation
    ├── audio                     # Compressed audio (includes text)
    ├── state
    │   ├── left_hand
    │   │   ├── encoder
    │   │   │   ├── data
    │   │   │   └── timestamp
    │   │   └── tactile
    │   │       ├── data
    │   │       └── timestamp
    │   └── right_hand
    │       ├── encoder
    │       │   ├── data
    │       │   └── timestamp
    │       └── tactile
    │           ├── data
    │           └── timestamp
    └── image
        ├── rgbd_rgb_extrinsic
        ├── RGB_CameraXXX/
        │   ├── timestamp
        │   ├── data
        │   ├── extrinsics
        │   └── intrinsics
        └── RGBD_CameraXXX/
            ├── timestamp
            ├── extrinsics
            ├── inner_extrinsics
            ├── color/
            │   ├── data
            │   └── intrinsics
            ├── depth/
            │   └── data
            ├── left/
            │   ├── data
            │   └── intrinsics
            └── right/
                ├── data
                └── intrinsics
......


```

## DF-2: First Output
DF-2 adds pose estimation results, parses tactile/encoder streams, and expresses camera extrinsics in a unified world frame (with `RGBD_0` as the origin).
### Filename
```text
episode_{INDEX}_{HHMMSS}_{ROOM-ID}_{PERSONNEL-ID}_glove.hdf5
```

### HDF5 structure
```text
/dataset
├── attributes                    # e.g., generated_time, data_id (compressed error info)
├── action                        # Action signals (no tactile)
│   ├── lefthand
│   │   ├── attributes            # description, etc.
│   │   ├── joints
│   │   │   ├── data              # (n, 29) joint angles in URDF joint order
│   │   │   └── attributes        # joint_names = [...]
│   │   └── handpose
│   │       ├── data              # (n, 7)
│   │       └── attributes        # order = [x, y, z, qw, qx, qy, qz]
│   └── righthand
│       ├── attributes            # description, hand_name, urdf, etc.
│       ├── joints
│       │   ├── data              # (n, 29)
│       │   └── attributes        # joint_names = [...]
│       └── handpose
│           ├── data              # (n, 7)
│           └── attributes        # order = [x, y, z, qw, qx, qy, qz]
└── observation                   # Episode state
    ├── audio                     # Compressed audio stream (includes text)
    ├── image
    │   ├── RGB_CameraXXX
    │   │   ├── data              # 1D compressed payload
    │   │   ├── extrinsics
    │   │   └── intrinsics        # attrs include width/height
    │   ├── RGBD_XXX
    │   │   ├── data              # 1D compressed payload
    │   │   ├── extrinsics
    │   │   ├── intrinsics
    │   │   └── attributes        # width/height
    │   └── [...]
    ├── lefthand
    │   ├── attributes            # description, etc.
    │   ├── joints
    │   │   ├── data              # (n, 29)
    │   │   └── attributes        # joint_names = [...]
    │   ├── handpose
    │   │   ├── data              # (n, 7)
    │   │   └── attributes        # order = [x, y, z, qw, qx, qy, qz]
    │   └── tactile
    │       ├── data              # (n, 3465)
    │       └── attributes        # sensor_names, sensor_lengths, etc.
    ├── righthand
    │   ├── attributes
    │   ├── joints
    │   │   ├── data              # (n, 29)
    │   │   └── attributes
    │   ├── handpose
    │   │   ├── data              # (n, 7)
    │   │   └── attributes
    │   └── tactile
    │       ├── data              # (n, 3465)
    │       └── attributes
    ├── obj1
    │   ├── data                  # (n, 17)
    │   └── attributes            # obj_name, obj_id, order/detail
    ├── obj2
    └── [...]

```

## DF-2R: Second Output
DF-2R is produced by retargeting DF-2 to a dexterous hand model. Supported models: **MANO**, **DexH13**, **DexH5**.

### Filename
```text
episode_{INDEX}_{HHMMSS}_{ROOM-ID}_{PERSONNEL-ID}_{HAND_MODEL}.hdf5
```

### HDF5 structure
**Key differences:**
- Hand joint dimension becomes 16 for DexH13/DexH5 examples.               
- Hand attributes describe the retargeted model/URDF.
```text
/dataset
├── attributes                    # e.g., generated_time, data_id (compressed error info)
├── action                        # Action signals (no tactile)
│   ├── lefthand
│   │   ├── attributes            # description, etc.
│   │   ├── joints
│   │   │   ├── data              # (n, 17) joint angles in URDF joint order
│   │   │   └── attributes        # joint_names = [...]
│   │   └── handpose
│   │       ├── data              # (n, 7)
│   │       └── attributes        # order = [x, y, z, qw, qx, qy, qz]
│   └── righthand
│       ├── attributes            # description, hand_name, urdf, etc.
│       ├── joints
│       │   ├── data              # (n, 17)
│       │   └── attributes        # joint_names = [...]
│       └── handpose
│           ├── data              # (n, 7)
│           └── attributes        # order = [x, y, z, qw, qx, qy, qz]
└── observation                   # Episode state
    ├── audio                     # Compressed audio stream (includes text)
    ├── image
    │   ├── RGB_CameraXXX
    │   │   ├── data              # 1D compressed payload
    │   │   ├── extrinsics
    │   │   └── intrinsics        # attrs include width/height
    │   ├── RGBD_XXX
    │   │   ├── data              # 1D compressed payload
    │   │   ├── extrinsics
    │   │   ├── intrinsics
    │   │   └── attributes        # width/height
    │   └── [...]
    ├── lefthand
    │   ├── attributes            # description, etc.
    │   ├── joints
    │   │   ├── data              # (n, 17)
    │   │   └── attributes        # joint_names = [...]
    │   ├── handpose
    │   │   ├── data              # (n, 7)
    │   │   └── attributes        # order = [x, y, z, qw, qx, qy, qz]
    │   └── tactile
    │       ├── data              # (n, 3750)
    │       └── attributes        # sensor_names, sensor_lengths, etc.
    ├── righthand
    │   ├── attributes
    │   ├── joints
    │   │   ├── data              # (n, 17)
    │   │   └── attributes
    │   ├── handpose
    │   │   ├── data              # (n, 7)
    │   │   └── attributes
    │   └── tactile
    │       ├── data              # (n, 3750)
    │       └── attributes
    ├── obj1
    │   ├── data                  # (n, 17)
    │   └── attributes            # obj_name, obj_id, order/detail
    ├── obj2
    └── [...]
```


### State and Action in DF-2 & DF-2R
The `observation` group in DF-2 and DF-2R is the **state** data in the episode, while `action` is one frame behind `observation`. To ensure both arrays have equal length, the last frame in `action` is repeated. Their mappings to timestamps and states are as follows:
| **Frame id** | $$1$$ | $$2$$ | ...... | $$n-1$$ | $$n$$ |
|----------|----------|---------|---------|---------|---------|
| **Observation** | $$S_1$$ | $$S_2$$ |  ...... | $$S_{n-1}$$ | $$S_n$$ |
| **Action** | $$S_2$$ | $$S_3$$ |  ...... | $$S_n$$ | $$S_n$$ |

## DF-3: Third Output
DF-3 is produced by converting DF-2R to **LeRobot Dataset v2.1** format.        
The official [repo](https://github.com/huggingface/lerobot) provide scripts to convert v2.1 to v3.0.       

# Distinguish Between Different Stages/Formats by Suffix
DF-3 is distinct because it is stored as a LeRobot Dataset. All other stages are HDF5 files and can be identified by suffix:      
| Stage | Suffix | Example |
|----------|----------| -----------------|
| **DF-1** | No suffix | episode_11_110000_111_100000.hdf5 |
| **DF-2** | "_glove" | episode_11_110000_111_100000_glove.hdf5 |
| **DF-2R** | "_{MODEL}" | episode_11_110000_111_100000_mano.hdf5 <br> episode_11_110000_111_100000_dh13.hdf5 <br> episode_11_110000_111_100000_dh5.hdf5|