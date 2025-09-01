import os
import sys
import argparse
import numpy as np


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default=f'{code_dir}/../assets/left.png', type=str)
    args = parser.parse_args()
    source_dir = args.source_dir

    for episode_dir in os.scandir(source_dir):
        if not episode_dir.is_dir():
            continue
        # 遍历每个episode文件夹
        for cam_dir in os.scandir(episode_dir.path):
            if not cam_dir.is_dir():
                continue
            # 遍历每个相机文件夹
            cam_name = cam_dir.name
            if not cam_name.startswith("RGBD_"):
                continue
            intrinsic_file = os.path.join(cam_dir.path, f"{cam_name}_left_intrinsic.txt")
            if not os.path.isfile(intrinsic_file):
                raise FileNotFoundError(f"Intrinsic file {intrinsic_file} not found in {cam_dir.path}")
            # 读取并验证内参矩阵
            matrix = []
            with open(intrinsic_file, 'r') as f:
                for line in f:
                    # 清理数据并分割为数值
                    elements = line.strip().split()
                    if len(elements) != 3:
                        raise ValueError(f"文件格式错误: 每行应有3个数值，实际得到 {len(elements)} 个")
                    matrix.extend(elements)
            # 验证矩阵维度
            if len(matrix) != 9:
                raise ValueError(f"内参矩阵应为3x3 (9个元素)，实际得到 {len(matrix)} 个元素")

            # 构建输出文件路径
            output_file = os.path.join(cam_dir.path, "foundationstereo_intrinsic.txt")

            # 写入新文件
            with open(output_file, 'w') as f:
                # 写入一行格式的内参矩阵
                f.write(" ".join(matrix) + "\n")
                # 写入第二行的固定值
                f.write("0.095\n")

            print(f"成功生成文件: {output_file}")






