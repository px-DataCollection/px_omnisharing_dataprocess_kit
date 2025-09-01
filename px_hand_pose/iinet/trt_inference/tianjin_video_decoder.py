import json
import os
import subprocess

import cv2
import numpy as np


class TianJinVideoDecoder:
    def __init__(self, data_root, color=True, left=True, right=True, depth=True, cam_baseline=0.095, frame_size=(1280, 720)):
        self.data_root = data_root
        cam_name = os.path.basename(self.data_root)

        self.W, self.H = frame_size

        self.K_color = np.loadtxt(f'{self.data_root}/{cam_name}_color_intrinsic.txt').reshape(3,3)
        self.K_depth = np.loadtxt(f'{self.data_root}/{cam_name}_left_intrinsic.txt').reshape(3,3)
        self.depth2color = np.array(
            json.load(open(f'{self.data_root}/{cam_name}_inner_extrinsic.json'))[
                'left_to_color']).reshape((4, 4))
        self.color2depth = np.linalg.inv(self.depth2color)
        self.cam_baseline = cam_baseline

        self.color = color
        self.color_frame_ls = []
        self.rgb_dir = f'{self.data_root}/color'
        if color:
            # self.rgb_dir = f'{self.data_root}/color'
            # os.makedirs(self.rgb_dir, exist_ok=True)
            ffmpeg_color = [
                'ffmpeg',
                '-i', f'{self.data_root}/{cam_name}_color.mp4',  # 输入 H.265 视频
                '-f', 'rawvideo',  # 输出原始视频帧
                # '-pix_fmt', 'gray12le',  # 输出 16 位灰度格式
                '-pix_fmt', 'bgr24',  # 输出 16 位灰度格式
                '-'
            ]
            self.ffmpeg_color = subprocess.Popen(ffmpeg_color, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.rgb_frame_size = self.W * self.H * 3

        self.left = left
        self.left_frame_ls = []
        self.left_dir = f'{self.data_root}/left'
        if left:
            ffmpeg_left = [
                'ffmpeg',
                '-i', f'{self.data_root}/{cam_name}_left.mp4',  # 输入 H.265 视频
                '-f', 'rawvideo',  # 输出原始视频帧
                # '-pix_fmt', 'gray12le',  # 输出 16 位灰度格式
                '-pix_fmt', 'bgr24',  # 输出 16 位灰度格式
                '-'
            ]
            self.ffmpeg_left = subprocess.Popen(ffmpeg_left, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.left_frame_size = self.W * self.H * 3

        self.right = right
        self.right_frame_ls = []
        self.right_dir = f'{self.data_root}/right'
        if right:
            ffmpeg_right = [
                'ffmpeg',
                '-i', f'{self.data_root}/{cam_name}_right.mp4',  # 输入 H.265 视频
                '-f', 'rawvideo',  # 输出原始视频帧
                # '-pix_fmt', 'gray12le',  # 输出 16 位灰度格式
                '-pix_fmt', 'bgr24',  # 输出 16 位灰度格式
                '-'
            ]
            self.ffmpeg_right = subprocess.Popen(ffmpeg_right, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.right_frame_size = self.W * self.H * 3

        self.depth = depth
        self.depth_frame_ls = []
        self.depth_dir = f'{self.data_root}/depth'
        if depth:
            ffmpeg_depth = [
                'ffmpeg',
                '-i', f'{self.data_root}/{cam_name}_depth.mp4',  # 输入 H.265 视频
                '-f', 'rawvideo',  # 输出原始视频帧
                '-pix_fmt', 'gray12le',  # 输出 16 位灰度格式
                # '-pix_fmt', 'bgr24',  # 输出 16 位灰度格式
                '-'
            ]
            self.ffmpeg_depth = subprocess.Popen(ffmpeg_depth, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.depth_frame_size = self.W * self.H * 3

    def read_process(self, save=False):
        idx = 1
        if save:
            if self.color:
                os.makedirs(self.rgb_dir, exist_ok=True)
            if self.left:
                os.makedirs(self.left_dir, exist_ok=True)
            if self.right:
                os.makedirs(self.right_dir, exist_ok=True)
            if self.depth:
                os.makedirs(self.depth_dir, exist_ok=True)

        try:
            while True:
                # 读取一帧数据
                if self.color:
                    rgb_frame = self.ffmpeg_color.stdout.read(self.rgb_frame_size)
                    if len(rgb_frame) == 0:
                        break
                    rgb_frame = np.frombuffer(rgb_frame, dtype=np.uint8).reshape((self.H, self.W, 3))
                    self.color_frame_ls.append(rgb_frame)
                    if save:
                        cv2.imwrite(f'{self.rgb_dir}/{idx}.png', rgb_frame)

                if self.left:
                    left_frame = self.ffmpeg_left.stdout.read(self.left_frame_size)
                    if len(left_frame) == 0:
                        break
                    left_frame = np.frombuffer(left_frame, dtype=np.uint8).reshape((self.H, self.W, 3))
                    self.left_frame_ls.append(left_frame)
                    if save:
                        cv2.imwrite(f'{self.left_dir}/{idx}.png', left_frame)

                if self.right:
                    right_frame = self.ffmpeg_right.stdout.read(self.right_frame_size)
                    if len(right_frame) == 0:
                        break
                    right_frame = np.frombuffer(right_frame, dtype=np.uint8).reshape((self.H, self.W, 3))
                    self.right_frame_ls.append(right_frame)
                    if save:
                        cv2.imwrite(f'{self.right_dir}/{idx}.png', right_frame)

                if self.depth:
                    depth_frame = self.ffmpeg_depth.stdout.read(self.depth_frame_size)
                    if len(depth_frame) == 0:
                        break
                    depth_frame = np.frombuffer(depth_frame, dtype=np.uint16).reshape((self.H, self.W, 1))
                    depth_frame_final = depth_frame.copy()
                    depth_frame_final[depth_frame_final < 100] = 0
                    depth_frame_final[depth_frame_final > 4000] = 0
                    self.depth_frame_ls.append(depth_frame_final)
                    if save:
                        cv2.imwrite(f'{self.depth_dir}/{idx}.png', depth_frame_final)
                idx += 1
        except Exception as e:
            print(f"Error reading frames: {e}")
        finally:
            print(f"Processed {idx} frames.")
            if self.color:
                self.ffmpeg_color.stdout.close()
            if self.left:
                self.ffmpeg_left.stdout.close()
            if self.right:
                self.ffmpeg_right.stdout.close()
            if self.depth:
                self.ffmpeg_depth.stdout.close()
        return self.color_frame_ls, self.left_frame_ls, self.right_frame_ls, self.depth_frame_ls