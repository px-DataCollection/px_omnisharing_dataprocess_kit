import numpy as np
import cv2
import torch
from PIL import Image

def replace_zeros_with_neighbors(tensor):
    """
        Replace zero values in a 2D Tensor with the average of their non-zero neighbors.
        Parameters:
            tensor (torch.Tensor): The input 2D tensor (H, W)
        Returns:
            torch.Tensor: The tensor with zeros replaced by the average of their neighbors
        """
    device = tensor.device
    zero_mask = (tensor == 0)

    # 使用3D卷积核处理2D数据 (需要添加通道维度)
    kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32, device=device)
    kernel = kernel.view(1, 1, 3, 3)  # [out_ch, in_ch, H, W]

    # 添加padding和batch维度
    padded = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    padded = torch.nn.functional.pad(padded, (1, 1, 1, 1), mode='constant', value=0)

    # 计算邻居和与数量
    neighbor_sum = torch.nn.functional.conv2d(padded, kernel).squeeze()  # [H, W]
    neighbor_count = torch.nn.functional.conv2d((padded != 0).float(), kernel).squeeze()

    # 计算平均值并替换零值
    avg = torch.zeros_like(tensor)
    valid_mask = neighbor_count > 0
    avg[valid_mask] = neighbor_sum[valid_mask] / neighbor_count[valid_mask]

    result = tensor.clone()
    result[zero_mask] = avg[zero_mask]
    return result


def align_image_by_depth(depth, K_rgb, K_depth, rgb2depth, device, rgb=None, to_depth=True, fill_zero=False):
    '''
    Aligns an RGB image from camera A to camera B using depth information.
    Args:
        depth: torch.Tensor or numpy.ndarray, Depth frame from camera B (H, W), in meters.
        K_rgb: torch.Tensor or numpy.ndarray, 3x3 intrinsic matrix of camera A.
        K_depth: torch.Tensor or numpy.ndarray, 3x3 intrinsic matrix of camera B.
        rgb2depth: torch.Tensor or numpy.ndarray, 4x4 extrinsic transformation matrix from A to B.
        rgb: optional, torch.Tensor or numpy.ndarray, RGB image from camera A (H, W, 3).
            If it's a grayscale image, it will be expanded to 3 channels.
        to_depth: bool, if True, aligns RGB to depth frame, otherwise aligns depth to RGB.
        fill_zero: bool, if True, replaces zero values in the depth map with the mean of surrounding pixels.

    Returns:
        numpy.ndarray: Aligned RGB image if rgb is provided, otherwise aligned depth map.
    '''
    if isinstance(rgb, np.ndarray):
        rgb = torch.from_numpy(rgb).to(device)
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth).to(device)
    if isinstance(K_rgb, np.ndarray):
        K_rgb = torch.from_numpy(K_rgb).to(device).float()
    if isinstance(K_depth, np.ndarray):
        K_depth = torch.from_numpy(K_depth).to(device).float()
    if isinstance(rgb2depth, np.ndarray):
        rgb2depth = torch.from_numpy(rgb2depth).to(device).float()

    if rgb and len(rgb.shape) == 2:
        rgb = rgb.unsqueeze(-1).repeat(1, 1, 3)

    height, width = depth.shape

    # 生成像素坐标网格
    u, v = torch.meshgrid(torch.arange(width, device=device),
                          torch.arange(height, device=device), indexing='xy')
    pixels_b = torch.stack([u.reshape(-1), v.reshape(-1),
                            torch.ones_like(u.reshape(-1))], dim=0)

    # 3D投影计算
    depth_b_flat = depth.view(-1)
    points_3D_b = torch.linalg.inv(K_depth) @ (pixels_b * depth_b_flat)
    points_3D_b_h = torch.cat([points_3D_b, torch.ones(1, points_3D_b.shape[1], device=device)])
    points_3D_a_h = torch.linalg.inv(rgb2depth) @ points_3D_b_h
    points_3D_a = points_3D_a_h[:3]

    # 投影到RGB相机平面
    pixels_a_h = K_rgb @ points_3D_a
    pixels_a = pixels_a_h[:2] / pixels_a_h[2]

    if to_depth and rgb is not None:
        # 使用grid_sample进行双线性采样
        x_flat = (2.0 * pixels_a[0] / width - 1.0).view(height, width)
        y_flat = (2.0 * pixels_a[1] / height - 1.0).view(height, width)
        grid = torch.stack([x_flat, y_flat], dim=-1).unsqueeze(0)

        aligned_rgb = torch.nn.functional.grid_sample(
            rgb.permute(2, 0, 1).unsqueeze(0), grid,
            mode='bilinear', padding_mode='zeros', align_corners=False
        ).squeeze().permute(1, 2, 0)

        return aligned_rgb
    else:
        # 深度对齐优化
        valid_mask = (pixels_a[0] >= 0) & (pixels_a[0] < width) & \
                     (pixels_a[1] >= 0) & (pixels_a[1] < height)

        map_x = torch.round(pixels_a[0][valid_mask]).long().clamp(0, width - 1)
        map_y = torch.round(pixels_a[1][valid_mask]).long().clamp(0, height - 1)
        valid_depth = pixels_a_h[2][valid_mask]

        depth_a = torch.full((height, width), float('inf'),
                             dtype=torch.float32, device=device)

        # 使用scatter_reduce优化
        depth_a.view(-1).scatter_reduce_(
            0, map_y * width + map_x, valid_depth,
            reduce='amin', include_self=True
        )

        depth_a = torch.where(torch.isinf(depth_a),
                              torch.zeros((), device=device), depth_a)

        if fill_zero:
            depth_a = replace_zeros_with_neighbors(depth_a)

        return depth_a.cpu().numpy()

if __name__ == "__main__":

    # depth_map = torch.rand((720, 1280), dtype=torch.float32, device='cuda') * 10  # 模拟深度图
    K_rgb = torch.tensor([
        [1000, 0, 640],
        [0, 1000, 360],
        [0, 0, 1]
    ], dtype=torch.float32, device='cuda')
    K_depth = torch.tensor([
        [1000, 0, 640],
        [0, 1000, 360],
        [0, 0, 1]
    ], dtype=torch.float32, device='cuda')
    T_rgb_to_depth = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device='cuda')

    import time
    tt = time.time()
    for i in range(171):
        print("************************", i, "*************************")
        st = time.time()
        depth_map = torch.rand((720, 1280), dtype=torch.float32, device='cuda') * 10  # 模拟深度图
        align_image_by_depth(depth_map, K_rgb, K_depth, T_rgb_to_depth, 'cuda', fill_zero=True)
        print("Time taken for alignment: ", time.time() - st, "s")
    print("Total time for 1000 iterations: ", time.time() - tt, "s")
