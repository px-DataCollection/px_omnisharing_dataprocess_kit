import numpy as np
import torch
import trimesh


def depth_to_points(depth, K, R=None, t=None):
    """
    Reference: https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/utils/geometry.py
    """
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    if len(depth.shape) == 2:
        depth = depth[None]
    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]

    # from reference to target viewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    return pts3D_2[:, :, :, :3, 0][0]


def depth2pcd(depth, K, out_pth=None, rgb=None, R=None, t=None):
    """
    Convert depth map to point cloud.
    """
    points = depth_to_points(depth, K, R, t)
    if rgb is None:
        rgb = np.ones_like(points) * 255
    elif isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    points_1d = points.reshape(-1, 3)

    # Filter out points with depth == 0
    mask = points_1d[:, 2] > 0
    # mask = (points_1d[:, 2] > 0) & (points_1d[:, 2] < 2)
    # print("depth range:", np.unique(points_1d[:, 2]))
    points_1d = points_1d[mask]

    if out_pth:
        if rgb is not None:
            rgb_1d = rgb.reshape(-1, 3)
            rgb_1d = rgb_1d[mask]
            rgb_1d = rgb_1d.astype(np.uint8)
            trimesh.PointCloud(points_1d, rgb_1d).export(out_pth)
        else:
            trimesh.PointCloud(points_1d).export(out_pth)
    return points


def transform_pcd(pcd, RT):
    """
    Transform point cloud.
    :param
        pcd: (N, 3) numpy array
        RT: (4, 4) numpy array
    """
    pcd_h = np.hstack([pcd, np.ones((pcd.shape[0], 1))])
    pcd_h = RT @ pcd_h.T
    return pcd_h[:3].T


def filter_3D_points(pcd, ratio=0.1):
    """
    Filter out outliers from a 3D point cloud.

    :param pcd: (np.array) An (n, 3) numpy array representing the 3D point cloud.
    :param ratio: (float) The ratio of points considered as outliers.
                  For example, ratio=0.1 means the top 10% farthest points (outliers) will be removed.
    :return: (np.array) A filtered (m, 3) numpy array of inlier points (m <= n).
    """
    if not isinstance(pcd, np.ndarray) or pcd.shape[1] != 3:
        raise ValueError("Input pcd must be a numpy array of shape (n, 3)")

    if not (0 < ratio < 1):
        raise ValueError("Ratio must be between 0 and 1")

    # Compute the centroid (mean position) of the point cloud
    centroid = np.mean(pcd, axis=0)

    # Compute the Euclidean distance of each point from the centroid
    distances = np.linalg.norm(pcd - centroid, axis=1)

    # Determine the threshold distance for filtering out the top `ratio` outliers
    threshold = np.percentile(distances, (1 - ratio) * 100)

    # Select points that are within the threshold distance
    inliers = pcd[distances <= threshold]

    return inliers