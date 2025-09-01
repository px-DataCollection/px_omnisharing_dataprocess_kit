import os
import argparse
import shutil
import re

if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir',  type=str, required= True)
    parser.add_argument('--target_dir',  type=str, required= True)
    parser.add_argument('--source_camera', type=str, default="RGBD_0")
    args = parser.parse_args()
    source_dir = args.source_dir
    target_dir = args.target_dir

    obj_pose_dir = os.path.join(source_dir, 'obj_pose_results/ob_in_cam')
    right_bracelet_pose_dir = os.path.join(source_dir, 'right_bracelet_pose_results/ob_in_cam')
    left_bracelet_pose_dir = os.path.join(source_dir, 'left_bracelet_pose_results/ob_in_cam')
    color_dir = os.path.join(source_dir, 'color')

    target_obj_pose_dir = os.path.join(target_dir, 'aligned_obj_pose_results')
    target_right_bracelet_pose_dir = os.path.join(target_dir, 'coarse_right_bracelet_poses')
    target_left_bracelet_pose_dir = os.path.join(target_dir, 'coarse_left_bracelet_poses')
    target_color_dir = os.path.join(target_dir, 'aligned_color')
    
    os.makedirs(target_obj_pose_dir, exist_ok=True)
    os.makedirs(target_right_bracelet_pose_dir, exist_ok=True)
    os.makedirs(target_left_bracelet_pose_dir, exist_ok=True)
    os.makedirs(target_color_dir, exist_ok=True)

    # Copy timestamps
    if args.source_camera == "RGBD_0":
        timestamp_file = os.path.join(source_dir, 'RGBD_0_time_stamp.txt')
    elif args.source_camera == "RGBD_1":
        timestamp_file = os.path.join(source_dir, 'RGBD_1_time_stamp.txt')
    elif args.source_camera == "RGBD_2":
        timestamp_file = os.path.join(source_dir, 'RGBD_2_time_stamp.txt')
    else:
        print(args.source_camera)
        assert False, "Invalid RGBD Camera, must be 0, 1, 2"
    
    target_timestamp_file = os.path.join(target_dir, 'aligned_timestamp.txt')
    if os.path.exists(timestamp_file):
        shutil.copy2(timestamp_file, target_timestamp_file)
        print(f"Copied timestamps: {timestamp_file} → {target_timestamp_file}")
    else:
        print(f"File not found: {timestamp_file}")

    # Acquire all files from source directory
    for filename in os.listdir(color_dir):
        if not filename.endswith('.png'):
            continue

        # Check if the filename is integer
        match = re.match(r'^(\d+)\.png$', filename)
        if not match:
            continue

        # Read and validate the integer
        number_str = match.group(1)

        try:
            file_number = int(number_str)
        except ValueError:
            continue

        # Create a new filename
        new_filename = f"{file_number:06d}"

        # Construct complete file paths
        source_color_path = os.path.join(color_dir, number_str + '.png')
        target_color_path = os.path.join(target_color_dir, new_filename + '.png')

        source_obj_pose_path = os.path.join(obj_pose_dir, number_str + '.txt')
        target_obj_pose_path = os.path.join(target_obj_pose_dir, new_filename + '.txt')

        source_right_bracelet_pose_path = os.path.join(right_bracelet_pose_dir, number_str + '.txt')
        source_left_bracelet_pose_path = os.path.join(left_bracelet_pose_dir, number_str + '.txt')
    
        # Copy all files to the output path
        shutil.copy2(source_color_path, target_color_path)

        try: 
            shutil.copy2(source_right_bracelet_pose_path, target_right_bracelet_pose_dir)
        except FileNotFoundError:
            pass
        
        try: 
            shutil.copy2(source_left_bracelet_pose_path, target_left_bracelet_pose_dir)
        except FileNotFoundError:
            pass

