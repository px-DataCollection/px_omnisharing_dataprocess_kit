"""tactile_ForceFieldGernerator.py

触觉仿真数据插值模块，
输入： 每个指节一个正压力和力作用点的坐标
输出： 每个指节网格点的压力
算法: 高斯面插值，距离递减

"""

import torch
import numpy as np


class ForceFieldGenerator:
    """力场生成器类，处理高斯力场计算"""

    def __init__(self, device="cuda"):
        self.device = device

    def _ensure_tensor(self, data, dtype=torch.float32):
        """确保数据为tensor格式"""
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, dtype=dtype, device=self.device)
        return data.to(self.device)

    def generate_random_point(self, local_pos):
        """从局部坐标点云生成随机点"""
        local_pos = self._ensure_tensor(local_pos)
        ranges = torch.stack(
            [torch.min(local_pos, dim=0)[0], torch.max(local_pos, dim=0)[0]]
        )

        return torch.rand(3, device=self.device) * (ranges[1] - ranges[0]) + ranges[0]

    def calculate_force_field(
        self,
        local_pos,
        center_point,
        force_magnitude,
        sigma=None,
        transform_matrix=None,
        enhance_contrast=False,
    ):
        """计算高斯力场分布

        Args:
            local_pos: 局部坐标点云
            center_point: 力的作用中心点
            force_magnitude: 力的大小
            sigma: 高斯分布的标准差
            transform_matrix: 坐标转换矩阵
            enhance_contrast: 是否增强对比度

        Returns:
            torch.Tensor: 力场分布数据
        """
        # # 打印输入数据信息
        # print("Debug信息:")
        # print(f"local_pos type: {type(local_pos)}")
        # print(f"local_pos shape/len: {np.shape(local_pos) if isinstance(local_pos, np.ndarray) else len(local_pos)}")
        # print(f"center_point: {center_point}")
        # print(f"force_magnitude: {force_magnitude}")

        # 确保local_pos是正确的形状
        if isinstance(local_pos, list):
            local_pos = np.array(local_pos)
        if len(local_pos.shape) == 1:
            local_pos = local_pos.reshape(-1, 3)

        # 确保数据格式
        local_pos = self._ensure_tensor(local_pos)
        center_point = self._ensure_tensor(center_point)
        x_center, y_center = center_point

        # 计算距离
        dx = local_pos[0, :] - x_center
        dy = local_pos[1, :] - y_center
        distances = torch.sqrt(dx**2 + dy**2)

        # 计算sigma
        if sigma is None:
            sigma = torch.max(distances) / 5

        # 计算力场
        if enhance_contrast:
            # 使用更陡峭的函数增强对比度
            normalized_distances = (distances - distances.min()) / (
                distances.max() - distances.min()
            )
            Fz_interp = force_magnitude * (1 - normalized_distances) ** 20
        else:
            # 标准高斯分布
            Fz_interp = force_magnitude * torch.exp(-(distances**2) / (2 * sigma**2))

        return Fz_interp.tolist()
