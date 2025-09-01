import cv2
import pickle
import sys
import os
import pyrealsense2 as rs
import numpy as np
import time


pipeline = rs.pipeline()
config = rs.config()
# align = rs.align(rs.stream.color)

serial_number = None
if serial_number is None:
    context = rs.context()
    devices = context.query_devices()
    if len(devices) == 0:
        print("No device is connected")
    else:
        serial_number = devices[0].get_info(rs.camera_info.serial_number)

# 启用设备
config.enable_device(serial_number)
print('启动的摄像头序列号: {}'.format(serial_number))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 15)

# # Start streaming
cfg = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

color_sensor = cfg.get_device().query_sensors()[1]  # 彩色传感器
# # 设置手动曝光模式
color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)  # 关闭自动曝光
color_sensor.set_option(rs.option.exposure, 70.0)  # 设置曝光时间（单位：毫秒）
# color_sensor.set_option(rs.option.gain, 64.0)     # 补偿亮度

data_dir = 'none'

unaligned_depth_dir = os.path.join(data_dir, 'unaligned_depth')
os.makedirs(unaligned_depth_dir, exist_ok=True)
aligned_depth_dir = os.path.join(data_dir, 'depth')
os.makedirs(aligned_depth_dir, exist_ok=True)
unaligned_color_dir = os.path.join(data_dir, 'unaligned_rgb')
os.makedirs(unaligned_color_dir, exist_ok=True)
color_dir = os.path.join(data_dir, 'rgb')
os.makedirs(color_dir, exist_ok=True)
left_dir = os.path.join(data_dir, 'left')
os.makedirs(left_dir, exist_ok=True)
right_dir = os.path.join(data_dir, 'right')
os.makedirs(right_dir, exist_ok=True)

#开始采集数据
idx = 0
RecordStream = False
try:
    while True:
        # 等待并捕获一帧
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        unaligned_depth_frame = frames.get_depth_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()


        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)
        unaligned_color_frame = frames.get_color_frame()

        if not unaligned_depth_frame or not aligned_color_frame :
            continue

        # 获取图像数据并转换为NumPy数组
        unaligned_depth_image = np.asanyarray(unaligned_depth_frame.get_data())
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        unaligned_color_image = np.asanyarray(unaligned_color_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())

        left_image = np.asanyarray(left_frame.get_data())
        right_frame = np.asanyarray(right_frame.get_data())

        cv2.imshow('rgb', color_image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord(" "):
            if not RecordStream:
                time.sleep(0.2)
                RecordStream = True
                rgb_intrinsic = aligned_color_frame.profile.as_video_stream_profile().get_intrinsics()

                with open(os.path.join(data_dir, "cam_K.txt"), "w") as f:
                    f.write(f"{rgb_intrinsic.fx} {0.0} {rgb_intrinsic.ppx}\n")
                    f.write(f"{0.0} {rgb_intrinsic.fy} {rgb_intrinsic.ppy}\n")
                    f.write(f"{0.0} {0.0} {1.0}\n")

                left_intrinsic = left_frame.profile.as_video_stream_profile().get_intrinsics()
                with open(os.path.join(data_dir, "left_cam_K.txt"), "w") as f:
                    f.write(f"{left_intrinsic.fx} {0.0} {left_intrinsic.ppx} {0.0} {left_intrinsic.fy} {left_intrinsic.ppy} {0.0} {0.0} {1.0}\n")
                    f.write(f"{0.05}")
                print("Recording started")
            else:
                RecordStream = False
                print("Recording stopped")
        if RecordStream:
        # 构造文件名并保存图像
            unaligned_depth_file_name = os.path.join(unaligned_depth_dir, f'{idx}.png')
            aligned_depth_file_name = os.path.join(aligned_depth_dir, f'{idx}.png')
            unaligned_color_file_name = os.path.join(unaligned_color_dir, f'{idx}.png')
            color_file_name = os.path.join(color_dir, f'{idx}.png')
            left_file_name = os.path.join(left_dir, f'{idx}.png')
            right_file_name = os.path.join(right_dir, f'{idx}.png')

            cv2.imwrite(unaligned_depth_file_name, unaligned_depth_image)
            cv2.imwrite(aligned_depth_file_name, aligned_depth_image)
            cv2.imwrite(unaligned_color_file_name, unaligned_color_image)
            cv2.imwrite(color_file_name, color_image)
            cv2.imwrite(left_file_name, left_image)
            cv2.imwrite(right_file_name, right_frame)
            print(f'保存第{idx}帧')
            idx += 1
        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    # 停止管道
    pipeline.stop()

