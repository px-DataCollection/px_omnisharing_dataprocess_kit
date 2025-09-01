import argparse
import os
import time

import cv2
import numpy as np
import torch
import pycuda.driver as cuda

from depth_estimation import IINetTRT
from tianjin_video_decoder import TianJinVideoDecoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='none')
    parser.add_argument('--depth_save_dir', type=str, default='aligned_depth', help='Directory to save depth images')
    parser.add_argument('--engine_path', type=str, default='none',
                        help='Path to the TensorRT engine file')
    parser.add_argument('--cam_baseline', type=float, default=0.095, help='camera baseline')
    parser.add_argument('--frame_shape', type=int, nargs=2, default=[1280, 720],
                        help='Frame shape of the video (width, height)')

    args = parser.parse_args()

    torch.cuda.init()
    # 自动选择当前可用设备
    device_id = torch.cuda.current_device()
    device = torch.device(f'cuda:{device_id}')

    # 使用PyTorch管理的当前设备
    ctx = cuda.Device(device_id).retain_primary_context()
    ctx.push()

    IINet = IINetTRT(
        engine_path=args.engine_path,
        ratio=1.0,
    )

    load_data_time = time.time()
    data_root = args.data_dir
    video_decoder = TianJinVideoDecoder(
        data_root,
        color=False,
        left=True,
        right=True,
        depth=False,
        cam_baseline=args.cam_baseline,
        frame_size=args.frame_shape,
    )

    K_color = video_decoder.K_color
    K_depth = video_decoder.K_depth
    color2depth = video_decoder.color2depth
    baseline = args.cam_baseline

    color_frame_ls, left_frame_ls, right_frame_ls, depth_frame_ls = video_decoder.read_process(save=False)
    H, W = video_decoder.H, video_decoder.W
    print("Load data time: ", time.time() - load_data_time, "s")

    st2 = time.time()
    depth_data = []
    for i in range(len(left_frame_ls)):
        print("****************************", i, "****************************")
        t_frame = time.time()
        depth = IINet.predict(
            left_frame_ls[i],
            right_frame_ls[i],
            K_color,
            K_depth,
            baseline,
            color2depth,
            align=True,
        )
        t_move = time.time()
        depth_data.append(depth)
        print("Move to CPU time: ", time.time() - t_move, "s")
        print("Total time for frame {}: {}".format(i, time.time() - t_frame))

    torch.cuda.empty_cache()
    ctx.pop()
    print("- " *50)
    print("Calculate depth time taken: ", time.time() - st2)

    st3 = time.time()
    depth_save_dir = os.path.join(data_root, args.depth_save_dir)
    os.makedirs(depth_save_dir, exist_ok=True)
    j = 1
    for depth in depth_data:
        cv2.imwrite(f"{depth_save_dir}/{j}.png", depth)
        j += 1
    print("Save depth time taken: ", time.time() - st3)

