import cv2
import pickle
import sys
import os
# import pyrealsense2 as rs
import numpy as np
import argparse
import json

def align_and_save_depth(depth_img, K_depth, K_rgb, R, T):
    """
    深度图对齐并保存为16位毫米单位深度图
    :param depth_img: 原始深度图(uint16, 单位mm)
    :param K_depth: 深度相机内参3x3
    :param K_rgb: RGB相机内参3x3
    :param R: 旋转矩阵3x3 (深度→RGB)
    :param T: 平移向量3x1 (深度→RGB)
    :param output_path: 输出路径
    """
    h_depth, w_depth = depth_img.shape
    h_rgb, w_rgb = depth_img.shape  # rgb和深度图像尺寸相同

    T = T.reshape(3, 1)

    # 生成深度图像素网格
    y_depth, x_depth = np.indices((h_depth, w_depth))
    pixels_depth = np.stack([x_depth.ravel(), y_depth.ravel(),
                             np.ones_like(x_depth.ravel())], axis=1).T

    # 反投影到深度相机坐标系（保持毫米单位）
    points_3d_depth = np.linalg.inv(K_depth) @ pixels_depth
    points_3d_depth *= depth_img.ravel()[None, :]  # 单位保持毫米

    print(f"R shape: {R.shape}")  # 应输出 (3, 3)
    print(f"points_3d_depth shape: {points_3d_depth.shape}")  # 应输出 (3, N)
    print(f"T shape: {T.shape}")  # 调整后应输出 (3, 1)

    # 坐标系转换到RGB相机
    points_3d_rgb = R @ points_3d_depth + T

    # 投影到RGB图像平面
    pixels_rgb_homo = K_rgb @ points_3d_rgb
    pixels_rgb = (pixels_rgb_homo[:2] / pixels_rgb_homo[2]).T

    # 初始化对齐后的深度图
    aligned_depth = np.zeros((h_rgb, w_rgb), dtype=np.uint16)

    # 坐标有效性判断
    x_rgb = pixels_rgb[:, 0].clip(0, w_rgb - 1).astype(int)
    y_rgb = pixels_rgb[:, 1].clip(0, h_rgb - 1).astype(int)
    valid = (x_rgb >= 0) & (x_rgb < w_rgb) & (y_rgb >= 0) & (y_rgb < h_rgb)

    # 填充有效深度值
    aligned_depth[y_rgb[valid], x_rgb[valid]] = depth_img.ravel()[valid]
    return aligned_depth

if __name__ == "__main__":
    ## 将文件夹中所有深度图对齐到对应的彩色图保存
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir',  type=str, required= True)
    args = parser.parse_args()
    source_dir = args.source_dir

    cam_name = source_dir[source_dir.rfind('/') + 1:]
    inner_extrinsic_file = os.path.join(source_dir, cam_name+'_inner_extrinsic.json')
    with open(inner_extrinsic_file, 'r') as f:
        inner_extrinsic_data = json.load(f)
        depth2color_extrinsic = np.array(inner_extrinsic_data['left_to_color'])

    depth2color_R = depth2color_extrinsic[:3, :3]
    depth2color_T = depth2color_extrinsic[:3, 3]*1000  # 转换为毫米单位

    rgb_intrinsic_file = os.path.join(source_dir, cam_name+'_color_intrinsic.txt')
    rgb_intrinsic = np.loadtxt(rgb_intrinsic_file, delimiter=' ')

    depth_intrinsic_file = os.path.join(source_dir, cam_name+'_left_intrinsic.txt')
    depth_intrinsic = np.loadtxt(depth_intrinsic_file, delimiter=' ')


    unaligned_depth_folder = os.path.join(source_dir, 'unaligned_depth')
    rgb_folder = os.path.join(source_dir, 'color')
    aligned_depth_folder = os.path.join(source_dir, 'aligned_depth')

    os.makedirs(aligned_depth_folder, exist_ok=True)

    for root, dirs, files in os.walk(unaligned_depth_folder):
        for file in files:
            if file.endswith('.png'):
                depth_file = os.path.join(unaligned_depth_folder, file)
                rgb_file = os.path.join(rgb_folder, file)
                aligned_depth_file = os.path.join(aligned_depth_folder, file)
                depth_image = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
                rgb_image = cv2.imread(rgb_file)
                aligned_depth = align_and_save_depth(depth_image, depth_intrinsic, rgb_intrinsic, depth2color_R, depth2color_T)
                cv2.imwrite(aligned_depth_file, aligned_depth)
                print(f"Save aligned depth to {aligned_depth_file}")