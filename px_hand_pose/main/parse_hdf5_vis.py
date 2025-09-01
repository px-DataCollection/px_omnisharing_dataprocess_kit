import h5py
import os
import tempfile
from PIL import Image
import numpy as np
import json
import shutil
import argparse

def parse_processed_h5(
        processed_h5_file: str,
        results_dir: str,
        return_vision: bool = True,
):
    returns = {}
    f = h5py.File(processed_h5_file, 'r')
    if return_vision:
        image_group = f["dataset/observation/image"]

        #rgb_rgbd_extrinsic
        os.makedirs(results_dir, exist_ok=True)
        rgbd_rgb_extrinsic = image_group["rgbd_rgb_extrinsic"][:]
        rgbd_rgb_extrinsic_file = os.path.join(results_dir, "rgbd_rgb_extrinsic.json")
        json_str = rgbd_rgb_extrinsic.item().decode('utf-8')
        json_data = json.loads(json_str)
        with open(rgbd_rgb_extrinsic_file, "w", encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)  # 保持非ASCII字符原样
        for video_name in image_group.keys():
            print('video_name', video_name)
            if video_name.startswith("RGBD_"):
                video_dir = os.path.join(results_dir,video_name)
                os.makedirs(video_dir,exist_ok = True)

                #right 相机
                if 'data' not in image_group[f"{video_name}/right"].keys():
                    # 删除 results_dir
                    print(f"Skipping {video_name} right camera data not found")
                    shutil.rmtree(results_dir, ignore_errors=True)
                    return  None

                video_data = image_group[f"{video_name}/right/data"][:]
                video_file = os.path.join(video_dir, f"{video_name}_right.mp4")
                with open(video_file, "wb") as f:
                    f.write(video_data)

                # color 相机
                video_data = image_group[f"{video_name}/color/data"][:]
                video_file = os.path.join(video_dir, f"{video_name}_color.mp4")
                with open(video_file, "wb") as f:
                    f.write(video_data)
                color_intrinsic_data = image_group[f"{video_name}/color/intrinsics"][:]
                color_intrinsic_data = np.array(color_intrinsic_data).reshape(3, 3)
                color_intrinsic_file = os.path.join(video_dir, f"{video_name}_color_intrinsic.txt")
                np.savetxt(color_intrinsic_file, color_intrinsic_data, fmt='%.6f', delimiter=' ')


                # left 相机
                video_data = image_group[f"{video_name}/left/data"][:]
                video_file = os.path.join(video_dir, f"{video_name}_left.mp4")
                with open(video_file, "wb") as f:
                    f.write(video_data)
                left_intrinsic_data = image_group[f"{video_name}/left/intrinsics"][:]
                left_intrinsic_data = np.array(left_intrinsic_data).reshape(3, 3)
                left_intrinsic_file = os.path.join(video_dir, f"{video_name}_left_intrinsic.txt")
                np.savetxt(left_intrinsic_file, left_intrinsic_data, fmt='%.6f', delimiter=' ')

                # depth 相机
                try:
                    video_data = image_group[f"{video_name}/depth/data"][:]
                    video_file = os.path.join(video_dir, f"{video_name}_depth.mp4")
                    with open(video_file, "wb") as f:
                        f.write(video_data)
                except Exception as e:
                    print("K, no depth camera")

                # 将 time_stamp_data保存成txt文件
                time_stamp_data = image_group[f"{video_name}/timestamp"][:]
                time_stamp_file = os.path.join(video_dir, f"{video_name}_time_stamp.txt")
                with open(time_stamp_file, "w") as f:
                    for time_stamp in time_stamp_data:
                        f.write(f"{time_stamp}\n")

                # inner_extrinsic
                inner_extrinsic_data = image_group[f"{video_name}/inner_extrinsic"][:]
                inner_extrinsic_file = os.path.join(video_dir, f"{video_name}_inner_extrinsic.json")
                print("inner_extrinsic_data size ", inner_extrinsic_data.size)
                json_str = inner_extrinsic_data.item().decode('utf-8')
                json_data = json.loads(json_str)
                with open(inner_extrinsic_file, "w", encoding='utf-8') as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False)  # 保持非ASCII字符原样

                #extrinsic
                extrinsic_data = image_group[f"{video_name}/extrinsics"][:]
                extrinsic_data = np.array(extrinsic_data).reshape(4, 4)
                extrinsic_file = os.path.join(video_dir, f"{video_name}_extrinsic.txt")
                np.savetxt(extrinsic_file, extrinsic_data, fmt='%.6f', delimiter=' ')

                print(video_name, video_file)
            elif video_name.startswith("RGB_"):
                if video_name == ('RGB_Camera5') or video_name == ('RGB_Camera7'):
                # 跳过 RGB_Camera5 和 RGB_Camera7
                    continue
                video_dir = os.path.join(results_dir, video_name)
                os.makedirs(video_dir, exist_ok=True)
                # rgb color 相机
                video_data = image_group[f"{video_name}/data"][:]
                video_file = os.path.join(video_dir, f"{video_name}_video.h265")
                with open(video_file, "wb") as f:
                    f.write(video_data)

                extrinsic_data = image_group[f"{video_name}/extrinsics"][:]
                extrinsic_data = np.array(extrinsic_data).reshape(4, 4)
                extrinsic_data_file = os.path.join(video_dir, f"{video_name}_extrinsic.txt")
                np.savetxt(extrinsic_data_file, extrinsic_data, fmt='%.6f', delimiter=' ')

                intrinsic_data = image_group[f"{video_name}/intrinsics"][:]
                intrinsic_data = np.array(intrinsic_data).reshape(3, 3)
                intrinsic_data_file = os.path.join(video_dir, f"{video_name}_intrinsic.txt")
                np.savetxt(intrinsic_data_file, intrinsic_data, fmt='%.6f', delimiter=' ')

    return returns


if __name__ == "__main__":

    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir',  type=str, required= True)
    parser.add_argument('--results_dir', type=str, required= True)
    args = parser.parse_args()
    source_dir = args.source_dir
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)


    for filename in os.listdir(source_dir):
        if filename.endswith('.hdf5'):
            # 如果在results_dir中已经存在同名文件夹，则跳过
            basename = os.path.splitext(filename)[0]
            target_dir = os.path.join(results_dir, basename)
            if os.path.exists(target_dir):
                print(f"File {filename} already exists in {results_dir}, skipping.")
                continue
            print('processing', filename)
            # basename = os.path.splitext(filename)[0]
            # target_dir = os.path.join(results_dir, basename)
            os.makedirs(target_dir,exist_ok = True)
            h5_file = os.path.join(source_dir,filename)
            returns = parse_processed_h5(
                h5_file,
                target_dir,
                return_vision=True,
            )