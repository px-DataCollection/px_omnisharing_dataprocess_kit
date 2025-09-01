import argparse
import os
import torch
import json
import cv2
import sys
import numpy as np
from typing import List
import trimesh
from scipy.spatial.transform import Rotation
from utils.kalman_filter_6d import KalmanFilter6D
from utils.visualizetion import draw_bbox

src_path = os.path.join(os.path.dirname(__file__), "..")
foundationpose_path = os.path.join(src_path, "FoundationPose")
if src_path not in sys.path:
    sys.path.append(src_path)
if foundationpose_path not in sys.path:
    sys.path.append(foundationpose_path)

from yolo import yolo
from FoundationPose.estimater import (
    ScorePredictor,
    PoseRefinePredictor,
    dr,
    FoundationPose,
    draw_xyz_axis
)
import logging


def get_sorted_frame_list(dir: str) -> List:
    files = os.listdir(dir)
    if not files:
        return []
    files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    if not files:
        return []
    if files[0].count('.') == 1:
        files.sort(key=lambda x: int(x.split('.')[0]))
    elif files[0].count('.') == 2:
        files.sort(key=lambda x: int(x.split('.')[0] + x.split('.')[1]))
    return files


def adjust_pose_to_image_point(
        ob_in_cam: torch.Tensor,
        K: torch.Tensor,
        x: float = -1.,
        y: float = -1.,
) -> torch.Tensor:
    """
    Adjusts the 6D pose(s) so that the projection matches the given 2D coordinate (x, y).

    Parameters:
    - ob_in_cam: Original 6D pose(s) as [4,4] or [B,4,4] tensor.
    - K: Camera intrinsic matrix (3x3 tensor).
    - x, y: Desired 2D coordinates on the image plane.

    Returns:
    - ob_in_cam_new: Adjusted pose(s) in same shape as input (tensor).
    """
    device = ob_in_cam.device
    dtype = ob_in_cam.dtype

    is_batched = ob_in_cam.ndim == 3
    if not is_batched:
        ob_in_cam = ob_in_cam.unsqueeze(0)  # [1, 4, 4]

    B = ob_in_cam.shape[0]
    ob_in_cam_new = torch.eye(4, device=device, dtype=dtype).repeat(B, 1, 1)

    for i in range(B):
        R = ob_in_cam[i, :3, :3]
        t = ob_in_cam[i, :3, 3]

        tx, ty = get_pose_xy_from_image_point(ob_in_cam[i], K, x, y)
        t_new = torch.tensor([tx, ty, t[2]], device=device, dtype=dtype)

        ob_in_cam_new[i, :3, :3] = R
        ob_in_cam_new[i, :3, 3] = t_new

    return ob_in_cam_new if is_batched else ob_in_cam_new[0]


def get_pose_xy_from_image_point(
        ob_in_cam: torch.Tensor,
        K: torch.Tensor,
        x: float = -1.,
        y: float = -1.,
) -> tuple:
    """
    Computes new (tx, ty) in camera space such that the projection matches image point (x, y).

    Parameters:
    - ob_in_cam: 4x4 pose tensor.
    - K: 3x3 intrinsic matrix tensor.
    - x, y: Desired image coordinates.

    Returns:
    - tx, ty: New x/y in camera coordinate system.
    """

    is_batched = ob_in_cam.ndim == 3
    if is_batched:
        ob_in_cam_new = ob_in_cam[0].cpu()  # [1, 4, 4]
    else:
        ob_in_cam_new = ob_in_cam.cpu()

    if x == -1. or y == -1.:
        return x, y

    t = ob_in_cam_new[:3, 3]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    tz = t[2]

    tx = (x - cx) * tz / fx
    ty = (y - cy) * tz / fy

    return tx, ty


def project_3d_to_2d(point_3d_homogeneous, K, ob_in_cam):
    # Transform point to camera frame
    point_cam = ob_in_cam @ point_3d_homogeneous

    # Perspective division to get normalized image coordinates
    x = point_cam[0] / point_cam[2]
    y = point_cam[1] / point_cam[2]

    # Apply camera intrinsics
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]

    return (int(u), int(v))


def get_mat_from_6d_pose_arr(pose_arr):
    # 提取位移 (xyz)
    xyz = pose_arr[:3]

    # 提取欧拉角
    euler_angles = pose_arr[3:]

    # 从欧拉角生成旋转矩阵
    rotation = Rotation.from_euler('xyz', euler_angles, degrees=False)
    rotation_matrix = rotation.as_matrix()

    # 创建 4x4 变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = xyz

    return transformation_matrix


def get_6d_pose_arr_from_mat(pose):
    if torch.is_tensor(pose):
        is_batched = pose.ndim == 3
        if is_batched:
            pose_np = pose[0].cpu().numpy()
        else:
            pose_np = pose.cpu().numpy()
    else:
        pose_np = pose

    xyz = pose_np[:3, 3]
    rotation_matrix = pose_np[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
    return np.r_[xyz, euler_angles]


def get_bbox_from_yolo(results, obj_name_list):
    bboxes = []
    for result in results:
        if result['name'] in obj_name_list:
            bboxes.append(result['bbox_cxywh'])
    return bboxes


def pose_track(
        rgb_seq_path: str,
        depth_seq_path: str,
        mesh_path: str,
        detect_model_path: str,
        init_mask_path: str,
        # cam_K: np.ndarray,
        cam_K_path: str,
        obj_name_list,
        pose_output_path: str,
        mask_visualization_path: str,
        bbox_visualization_path: str,
        pose_visualization_path: str,
        est_refine_iter: int,
        track_refine_iter: int,
        activate_2d_tracker: bool = False,
        activate_kalman_filter: bool = False,
):
    #################################################
    # Read the initial mask
    #################################################
    init_mask = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)
    if init_mask is None:
        print(f"Failed to read mask file {init_mask_path}.")
        return
    init_mask = init_mask.astype(bool)

    #################################################
    # Read the camera intrinsic matrix
    #################################################
    cam_K = np.loadtxt(cam_K_path).reshape(3, 3)

    #################################################
    # Read the frame list
    #################################################
    frame_color_list = get_sorted_frame_list(rgb_seq_path)
    frame_depth_list = get_sorted_frame_list(depth_seq_path)
    if not frame_color_list or not frame_depth_list:
        print(f"No RGB frames found.")
        return

    #################################################
    # Load the initial frame
    #################################################
    init_frame_filename = frame_color_list[0]
    init_frame_path = os.path.join(rgb_seq_path, init_frame_filename)
    init_frame = cv2.imread(init_frame_path)
    if init_frame is None:
        print(f"Failed to read initial frame.")
        return

    #################################################
    # Load the mesh
    #################################################
    from FoundationPose.estimater import trimesh_add_pure_colored_texture

    mesh_file = os.path.join(mesh_path)
    if not os.path.exists(mesh_file):
        print(f"Mesh file not found.")
        return
    mesh = trimesh.load(mesh_file)

    # if isinstance(mesh, trimesh.Scene):
    #     mesh = mesh.dump(concatenate=True)
    # # Convert units to meters
    # mesh.apply_scale(args.apply_scale)
    # if args.force_apply_color:
    #     mesh = trimesh_add_pure_colored_texture(mesh, color=np.array(args.apply_color), resolution=10)
    #
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    #################################################
    # Instantiate the 6D pose estimator
    #################################################




    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
    )
    logging.info("Estimator initialization done")

    #################################################
    # Instantiate the 2D tracker
    #################################################

    if activate_2d_tracker:
        obj_detector = yolo(model_path=detect_model_path, )

    #################################################
    # 6D pose tracking
    #################################################

    if activate_kalman_filter:
        kf = KalmanFilter6D(args.kf_measurement_noise_scale)

    total_frames = len(frame_color_list)
    pose_seq = [None] * total_frames  # Initialize as None
    kf_mean, kf_covariance = None, None

    os.makedirs(pose_visualization_path, exist_ok=True)
    ###########保存为视频
    output_file = f'{pose_visualization_path}/../pose.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_rate = 30
    frame_size = (1280, 720)
    video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

    # Forward processing from initial frame
    for i in range(0, total_frames):
        # print(f"Processing frame {i + 1}/{total_frames}...")
        #################################################
        # Read the frame
        #################################################
        frame_color_filename = frame_color_list[i]
        frame_depth_filename = frame_depth_list[i]
        # color = imageio.imread(os.path.join(rgb_seq_path, frame_color_filename))[..., :3]
        color = cv2.imread(os.path.join(rgb_seq_path, frame_color_filename))[..., :3]

        color = cv2.resize(color, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
        # print(f"Read color frame {frame_color_filename} with shape {color.shape}")

        depth = cv2.imread(os.path.join(depth_seq_path, frame_depth_filename), -1) / 1e3
        depth = cv2.resize(depth, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.001) | (depth >= np.inf)] = 0

        if color is None or depth is None:
            # print(f"Failed to read color frame {frame_color_filename} or depth map {frame_depth_filename}")
            continue
        #################################################
        # 6D pose tracking
        #################################################

        if i == 0:
            mask = init_mask.astype(np.uint8) * 255
            # cv2.imwrite('1.png',color)
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
            if activate_kalman_filter:
                kf_mean, kf_covariance = kf.initiate(get_6d_pose_arr_from_mat(pose))

            # pose 为4*4的矩阵
            if mask_visualization_path is not None:
                os.makedirs(mask_visualization_path, exist_ok=True)
            if bbox_visualization_path is not None:
                os.makedirs(bbox_visualization_path, exist_ok=True)
        else:
            if mask_visualization_path is not None:
                os.makedirs(mask_visualization_path, exist_ok=True)
            if bbox_visualization_path is not None:
                os.makedirs(bbox_visualization_path, exist_ok=True)

            ########## 2D tracking
            if activate_2d_tracker:
                #############将图片裁切成 960*704
                # 计算裁剪区域
                start_x, end_x = 160, 1120
                start_y, end_y = 8, 712
                # 执行裁剪（注意行列顺序：高度在前，宽度在后）
                cropped_img = color[start_y:end_y, start_x:end_x]

                yolo_results = obj_detector.predict(cropped_img, conf=0.6)
                bboxes = get_bbox_from_yolo(yolo_results, obj_name_list)
                if len(bboxes) != 1:
                    print(f'+++ Incorrect bracelet name! +++++++++++++@-@-@-@-@-@-@-@-@-@++++ yolo detect {len(bboxes)} boxes ++++++++++  +++++++++')
                    if activate_kalman_filter:
                        kf_mean, kf_covariance = kf.update(kf_mean, kf_covariance,
                                                           get_6d_pose_arr_from_mat(est.pose_last))
                else:
                    # 只支持一个mask
                    bboxes[0][0] += start_x
                    bboxes[0][1] += start_y
                    if activate_2d_tracker:
                        if not activate_kalman_filter:
                            est.pose_last = adjust_pose_to_image_point(ob_in_cam=est.pose_last, K=cam_K, x=bboxes[0][0],
                                                                       y=bboxes[0][1])
                        else:
                            # using kf to estimate the 6d estimation of the last pose
                            kf_mean, kf_covariance = kf.update(kf_mean, kf_covariance,
                                                               get_6d_pose_arr_from_mat(est.pose_last))
                            measurement_xy = np.array(
                                get_pose_xy_from_image_point(ob_in_cam=est.pose_last, K=cam_K, x=bboxes[0][0],
                                                             y=bboxes[0][1]))
                            kf_mean, kf_covariance = kf.update_from_xy(kf_mean, kf_covariance, measurement_xy)
                            est.pose_last = torch.from_numpy(get_mat_from_6d_pose_arr(kf_mean[:6])).unsqueeze(0).to(
                                est.pose_last.device)

            pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=track_refine_iter)
            if activate_2d_tracker and activate_kalman_filter:
                # use kf to predict from last pose, and update kf status
                kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)  # kf is alway one step behind

        # 保存pose.txt
        if not os.path.exists(pose_output_path):
            os.makedirs(pose_output_path, exist_ok=True)
        pose_filename = frame_color_filename.replace('.png', '.txt')
        pose_output_path_i = os.path.join(pose_output_path, pose_filename)
        np.savetxt(pose_output_path_i, pose.reshape(4, 4))

        if pose_visualization_path is not None:
            center_pose = pose
            vis_color = draw_bbox(color, ob_in_cam=center_pose, K=cam_K)
            vis_color = draw_xyz_axis(vis_color, ob_in_cam=center_pose, scale=0.05, K=cam_K, thickness=2,
                                      transparency=0,
                                      is_input_rgb=True)

            if not os.path.exists(pose_visualization_path):
                os.makedirs(pose_visualization_path, exist_ok=True)
            pose_visualization_color_filename = os.path.join(pose_visualization_path, frame_color_filename)
            cv2.imwrite(
                pose_visualization_color_filename, vis_color
            )
            video_writer.write(vis_color)

    # Clear GPU memory
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb_seq_path", type=str,
                        default="./color", help='Path to the RGB image sequence')
    parser.add_argument("--depth_seq_path", type=str,
                        default="./aligned_depth", help='Path to the depth image sequence')
    parser.add_argument("--mesh_path", type=str,
                        default=f"{src_path}/assets/data_project_bracelet_649/dataprojectbracelet649.obj")
    parser.add_argument("--detect_model_path", type=str,
                        default='./yolo.pt', help='Path to the YOLO detection model')
    parser.add_argument("--init_mask_path", type=str,
                        default="./obj_masks/1.png")
    parser.add_argument("--pose_output_path", type=str,
                        default="./obj_pose_refine_results")
    parser.add_argument("--mask_visualization_path", type=str, default=None)
    parser.add_argument("--bbox_visualization_path", type=str, default=None)
    parser.add_argument("--pose_visualization_path", type=str,
                        default="./obj_pose_refine_results/")
    parser.add_argument("--cam_K_path", type=str,
                        default="./cam_K.txt",
                        help="Camera intrinsic parameters")
    parser.add_argument('--obj_name_list',
                        nargs='+',  # 接受一个或多个参数
                        default=['bracelet_649'],
                        help='List of object names')
    parser.add_argument("--est_refine_iter", type=int, default=10,
                        help="FoundationPose initial refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--track_refine_iter", type=int, default=5,
                        help="FoundationPose tracking refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--activate_2d_tracker", type=bool, default=True, help="activate 2d tracker")
    parser.add_argument("--activate_kalman_filter", type=bool, default=True, help="activate kalman_filter")
    parser.add_argument("--kf_measurement_noise_scale", type=float, default=0.05,
                        help="The scale of measurement noise relative to prediction in kalman filter, greater value means more filtering. Only effective if activate_kalman_filter")
    parser.add_argument("--apply_scale", type=float, default=0.01,
                        help="Mesh scale factor in meters (1.0 means no scaling), commonly use 0.01")
    parser.add_argument("--force_apply_color", action='store_true', help="force a color for colorless mesh")
    parser.add_argument("--apply_color", type=json.loads, default="[0, 159, 237]",
                        help="RGB color to apply, in format 'r,g,b'. Only effective if force_apply_color")
    args = parser.parse_args()

    pose_track(
        args.rgb_seq_path,
        args.depth_seq_path,
        args.mesh_path,
        args.detect_model_path,
        args.init_mask_path,
        args.cam_K_path,
        args.obj_name_list,
        args.pose_output_path,
        args.mask_visualization_path,
        args.bbox_visualization_path,
        args.pose_visualization_path,
        args.est_refine_iter,
        args.track_refine_iter,
        args.activate_2d_tracker,
        args.activate_kalman_filter,
    )

    torch.cuda.empty_cache()
