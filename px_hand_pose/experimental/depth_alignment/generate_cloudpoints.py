import numpy as np
import cv2
import open3d as o3d


def create_point_cloud(rgb_path, depth_path, fx, fy, cx, cy, depth_scale=1000.0):
    # 读取图像
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(float)

    # 验证尺寸
    assert rgb.shape[:2] == depth.shape, "RGB和深度图尺寸不一致"

    # 创建像素坐标网格
    height, width = depth.shape
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)

    # 转换深度单位（假设深度图存储单位为毫米）
    z = depth

    # 计算三维坐标（向量化计算）
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    # 展平并过滤无效点（深度为0）
    valid_mask = z.flatten() > 0
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)[valid_mask]
    colors = rgb.reshape(-1, 3)[valid_mask] / 255.0

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def create_point_cloud_from_quantized_depth(quantized_depth, rgb_path,  fx, fy, cx, cy, depth_scale=1000.0):
    # 读取图像
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = quantized_depth

    # 验证尺寸
    assert rgb.shape[:2] == depth.shape, "RGB和深度图尺寸不一致"

    # 创建像素坐标网格
    height, width = depth.shape
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)

    # 转换深度单位（假设深度图存储单位为毫米）
    z = depth

    # 计算三维坐标（向量化计算）
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    # 展平并过滤无效点（深度为0）
    valid_mask = z.flatten() > 0
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)[valid_mask]
    colors = rgb.reshape(-1, 3)[valid_mask] / 255.0

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def create_compress_point_cloud(rgb_path, depth_path, fx, fy, cx, cy, depth_scale=1000.0):
    # 读取图像
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(float)
    depth = depth / 255.0 * 10000.0

    # 验证尺寸
    assert rgb.shape[:2] == depth.shape, "RGB和深度图尺寸不一致"

    # 创建像素坐标网格
    height, width = depth.shape
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)

    # 转换深度单位（假设深度图存储单位为毫米）
    z = depth

    # 计算三维坐标（向量化计算）
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    # 展平并过滤无效点（深度为0）
    valid_mask = z.flatten() > 0
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)[valid_mask]
    colors = rgb.reshape(-1, 3)[valid_mask] / 255.0

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def restore_depth(depth_16bit, quantized_8bit):
    restored = np.zeros_like(depth_16bit, dtype=np.float32)

    # 低区域反量化
    mask_low = quantized_8bit <= 31
    restored[mask_low] = quantized_8bit[mask_low] * (299 / 31)

    # 中区域反量化
    mask_mid = (quantized_8bit >= 32) & (quantized_8bit <= 223)
    restored[mask_mid] = 300 + (quantized_8bit[mask_mid] - 32) * (500 / 191)

    # 高区域反量化
    mask_high = quantized_8bit >= 224
    restored[mask_high] = 801 + (quantized_8bit[mask_high] - 224) * (699 / 31)
    return restored

k_fx = 644.196289
k_fy = 644.160095
k_cx = 638.465149
k_cy = 357.769867

# 使用示例
pcd = create_point_cloud(
    rgb_path="none",
    depth_path="none",
    fx=k_fx,
    fy=k_fy,
    cx=k_cx,
    cy=k_cy
)

# 可视化
# o3d.visualization.draw_geometries([pcd])
# 保存为PLY文件
o3d.io.write_point_cloud("none", pcd)
