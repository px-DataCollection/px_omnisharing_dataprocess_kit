##!/bin/bash

# PaXini Hand Pose (experimental)

# Install FoundationStereo on you Conda env before running the script

# Main script for generating 6D poses from raw data (Phase 1)
# Input: raw data from the EID factory, HDF5 format
# Output: 6D bracelet poses, text file format

set -euo pipefail

# 0. Initialization
if [ "$#" -ne 3 ]; then
  printf "Requires 3 arguments to run this script \n\n" 
  printf "Usage: $0 <hdf5_src_path> <depth_cam_index> <use_left_hand> \n"
  printf "Example: $0 /home/px/raw_data/input 0 false \n\n" 
  printf "The depth camera index must be 0, 1, 2; we recommend 0 \n" 
  printf "Assign false to <use_left_hand> if this episode does not contain left hand \n" 
  exit 1
fi

SOURCE_DIR="$1"
CAM_IND="$2"
USE_FEATURE="$3"

RGBD_A="paxini"

if [ "$CAM_IND" = "0" ]; then
  RGBD_A="RGBD_0"
elif [ "$CAM_IND" = "1" ]; then
  RGBD_A="RGBD_1"
elif [ "$CAM_IND" = "2" ]; then
  RGBD_A="RGBD_2"
else
  echo "Error: depth camera index must be 0, 1, or 2" >&2
  exit 1
fi

printf "==> [Step 0/2] EID processing initiates \n\n" 

## Set if the left hand should be screened
## The left hand is not visible in some collected data.
printf "Please ensure the right hand is visible in this episode, processing... \n\n" 
SCREEN_LH=false
if [ "$USE_FEATURE" = "true" ]; then
  printf "Please ensure the left hand is visible in this episode, processing... \n\n" 
elif [ "$USE_FEATURE" = "false" ]; then
  SCREEN_LH=true
else
  echo "Error: <use_left_hand> must be either true or false" >&2
  exit 1
fi

VIS_FINAL_DIR="$SOURCE_DIR/vis_res" # default path for bracelet pose results
VIS_INTMD_DIR="$SOURCE_DIR/debug" # default path for intermediate files, including parsed videos

cd "$(dirname "${BASH_SOURCE[0]}")" # >/dev/null 2>&1
SCRIPT_DIR="$(pwd -P)"
cd - >/dev/null 2>&1
printf "Estimating bracelet poses\n\n"
echo "Running from: $SCRIPT_DIR"

## Set the path to bracelet detection model
cd $SCRIPT_DIR
cd ../utils/model_both_bracelets/
BRACELET_MODEL="$(pwd -P)/detect_bracelet_648_649_v_0.0.1.pt"

# 1. Parse the input HDF5 file
cd $SCRIPT_DIR
~/anaconda3/envs/env_bracelet_2/bin/python parse_hdf5_vis.py --source_dir $SOURCE_DIR --results_dir $VIS_INTMD_DIR # > /dev/null 2>&1 


# 2. Decode recordings from RGBD cameras
export RGBD_A
find "$VIS_INTMD_DIR" -type d -name "$RGBD_A" -print0 |
xargs -0 -I {} sh -c '
    dir="$1"
    echo "Decoding dir: $dir" > /dev/null 2>&1 

    for subdir in color left right; do
        target="$dir/$subdir"
        if [ ! -d "$target" ]; then
            mkdir -p "$target" && echo "==> Created: $target" > /dev/null 2>&1 
        else
            echo "==> Found: $target" > /dev/null 2>&1 
        fi
    done

    cd "$dir" || exit 1
    for type in color left right; do
        video="${RGBD_A}_${type}.mp4"
        if [ -f "$video" ]; then
            ffmpeg -hide_banner -loglevel error -i "$video" "$type/%d.png"
        else
            echo "==> Video not found at: $dir/$video" 
        fi
    done
' _ {}

# 3. Run FoundationStereo for depth estimation
cd $SCRIPT_DIR
cd ../iinet
find "$VIS_INTMD_DIR" -type d -mindepth 1 -maxdepth 1 2>/dev/null | while read dir; do
    if [ -r "$dir" ]; then
        input_dir="$dir/$RGBD_A"
        out_dir="$dir/$RGBD_A/aligned_depth"
        engine_path="$SCRIPT_DIR/../iinet/checkpoints/iinet.engine"
        # echo "out_dir path: $out_dir" > /dev/null 2>&1 
        if [ ! -d "$out_dir" ]; then
            ~/anaconda3/envs/env_bracelet_1/bin/python  ./trt_inference/main.py --data_dir "$input_dir" \
                                  --engine_path "$engine_path"
        else
            echo "==> Found: $out_dir, skipped" > /dev/null 2>&1 
        fi
    else
        echo "Unauthorized at: $dir" >&2
    fi
done


# 4.1 Preparing data for FoundationPose: Right
cd $SCRIPT_DIR
cd ../FoundationPose/src
export CUDA_VISIBLE_DEVICES=0
find "$VIS_INTMD_DIR" -type d -mindepth 1 -maxdepth 1 2>/dev/null | while read dir; do
  if [ -r "$dir" ]; then
    foundationpose_source="$dir/$RGBD_A"
    cp "$foundationpose_source/${RGBD_A}_color_intrinsic.txt" "$foundationpose_source/cam_K.txt"

    ~/anaconda3/envs/env_bracelet_2/bin/python yolo_for_bracelet_mask.py --source_dir $foundationpose_source --model_path "$BRACELET_MODEL" --obj_name_list bracelet_649

  else
      echo "Unauthorized at: $dir" >&2
  fi
done

# 4.2 Preparing data for FoundationPose: Left
cd $SCRIPT_DIR
cd ../FoundationPose/src
find "$VIS_INTMD_DIR" -type d -mindepth 1 -maxdepth 1 2>/dev/null | while read dir; do
  if [ "$SCREEN_LH" = true ]; then
    echo "skipping left hand masking"
  elif [ -r "$dir" ]; then
    foundationpose_source="$dir/$RGBD_A"
    cp "$foundationpose_source/${RGBD_A}_color_intrinsic.txt" "$foundationpose_source/cam_K.txt"

    ~/anaconda3/envs/env_bracelet_2/bin/python yolo_for_bracelet_mask.py --source_dir $foundationpose_source --model_path "$BRACELET_MODEL" --obj_name_list bracelet_648
  else
      echo "Unauthorized at: $dir" >&2
  fi
done

# 5.1 Estimate bracelet pose via YOLO and FoundationPose: Right
cd $SCRIPT_DIR
cd ../FoundationPose/src
export CUDA_VISIBLE_DEVICES=0
find "$VIS_INTMD_DIR" -type d -mindepth 1 -maxdepth 1 2>/dev/null | while read dir; do
  if [ -r "$dir" ]; then
    foundationpose_source="$dir/$RGBD_A"
    echo "source for right hand: $foundationpose_source" > /dev/null 2>&1 
    ~/anaconda3/envs/env_bracelet_2/bin/python pose_estimator.py --mesh_path "../assets/data_project_bracelet_649/dataprojectbracelet649.obj" \
      --detect_model_path "$BRACELET_MODEL" \
      --rgb_seq_path "$foundationpose_source/color" \
      --depth_seq_path "$foundationpose_source/aligned_depth" \
      --init_mask_path "$foundationpose_source/right_bracelet_masks/1.png" \
      --pose_output_path "$foundationpose_source/right_bracelet_pose_results/ob_in_cam" \
      --pose_visualization_path "$foundationpose_source/right_bracelet_pose_results/track_vis" \
      --cam_K_path "$foundationpose_source/cam_K.txt" \
      --obj_name_list "['bracelet_649']" # > /dev/null 2>&1 
#                                 --debug_dir "$foundationpose_source/bracelet_pose_results"
  else
      echo "Unauthorized at: $dir" >&2
  fi
done

# 5.2 Estimate bracelet pose via YOLO and FoundationPose: Left
cd $SCRIPT_DIR
cd ../FoundationPose/src
export CUDA_VISIBLE_DEVICES=0
find "$VIS_INTMD_DIR" -type d -mindepth 1 -maxdepth 1 2>/dev/null | while read dir; do
  if [ "$SCREEN_LH" = true ]; then
    echo "skipping left hand pose"
  elif [ -r "$dir" ]; then
    foundationpose_source="$dir/$RGBD_A"
    echo "source for left hand: $foundationpose_source" > /dev/null 2>&1 
    ~/anaconda3/envs/env_bracelet_2/bin/python pose_estimator.py --mesh_path "../assets/data_project_bracelet_648/bracelet648.obj" \
      --detect_model_path "$BRACELET_MODEL" \
      --rgb_seq_path "$foundationpose_source/color" \
      --depth_seq_path "$foundationpose_source/aligned_depth" \
      --init_mask_path "$foundationpose_source/left_bracelet_masks/1.png" \
      --pose_output_path "$foundationpose_source/left_bracelet_pose_results/ob_in_cam" \
      --pose_visualization_path "$foundationpose_source/left_bracelet_pose_results/track_vis" \
      --cam_K_path "$foundationpose_source/cam_K.txt" \
      --obj_name_list "['bracelet_648']" # > /dev/null 2>&1 
#                                 --debug_dir "$foundationpose_source/bracelet_pose_results"
  else
      echo "Unauthorized at: $dir" >&2
  fi
done

# 6. Output pose files, and copy the raw HDF5 file
cd $SCRIPT_DIR
mkdir -p "$VIS_FINAL_DIR"
find "$VIS_INTMD_DIR" -type d -mindepth 1 -maxdepth 1 2>/dev/null | while read dir; do
  echo "Processing: $dir" > /dev/null 2>&1 
  if [ -r "$dir" ]; then
    source_dir="$dir/$RGBD_A"
    target_dir="$VIS_FINAL_DIR/$(basename "$dir")/$RGBD_A"
    ~/anaconda3/envs/env_bracelet_2/bin/python generate_glove_in.py --source_dir "$source_dir" \
                                  --target_dir "$target_dir" --source_camera "$RGBD_A" 
  else
      echo "Unauthorized at: $dir" >&2
  fi
done

cd $SCRIPT_DIR # smooth the bracelet poses
find "$VIS_INTMD_DIR" -type d -mindepth 1 -maxdepth 1 2>/dev/null | while read dir; do
  echo "Processing: $dir" > /dev/null 2>&1 
  if [ -r "$dir" ]; then
    target_dir="$VIS_FINAL_DIR/$(basename "$dir")/$RGBD_A"
    if [ "$SCREEN_LH" = false ]; then
      input_folder="$target_dir/coarse_left_bracelet_poses"
      output_folder="$target_dir/aligned_left_bracelet_pose_results" 
      ~/anaconda3/envs/foundationpose/bin/python pose_smooth_v3.py --input_folder "$input_folder" --output_folder "$output_folder"
    fi

    input_folder="$target_dir/coarse_right_bracelet_poses"
    output_folder="$target_dir/aligned_right_bracelet_pose_results" 
    ~/anaconda3/envs/foundationpose/bin/python pose_smooth_v3.py --input_folder "$input_folder" --output_folder "$output_folder"

  else
      echo "Unauthorized at: $dir" >&2
  fi
done

cd $SCRIPT_DIR
find "$VIS_FINAL_DIR" -type d -mindepth 1 -maxdepth 1 2>/dev/null | while read dir; do
  echo "Processing: $dir" > /dev/null 2>&1 
  if [ -r "$dir" ]; then
    source_dir="$SOURCE_DIR/"
    target_dir="$dir"
    cp -r "$source_dir/$(basename "$dir").hdf5" "$target_dir/"
  else
      echo "Unauthorized at: $dir" >&2
  fi
done

printf "==> [Step 1/2] Bracelet poses are generated.\n\n"
printf "Please run <px_retargeting> next\n"
