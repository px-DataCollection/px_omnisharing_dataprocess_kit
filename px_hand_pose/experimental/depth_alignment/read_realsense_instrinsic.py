import cv2
import pickle
import sys
import os
import pyrealsense2 as rs
import numpy as np

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

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

# config.enable_stream(rs.stream.color, 1280, 736, rs.format.bgr8, 15)
# # Start streaming
cfg = pipeline.start(config)
profile = cfg.get_stream(rs.stream.color)  # Fetch stream profile for color stream
intr = profile.as_video_stream_profile().get_intrinsics()
intr = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]])
print(intr)


#405
infrared_intr = cfg.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
infrared_intr = np.array([[infrared_intr.fx, 0, infrared_intr.ppx],
                          [0, infrared_intr.fy, infrared_intr.ppy],
                          [0, 0, 1]])

infrared_intr1 = cfg.get_stream(rs.stream.infrared, 2).as_video_stream_profile().get_intrinsics()
infrared_intr1 = np.array([[infrared_intr1.fx, 0, infrared_intr1.ppx],
                          [0, infrared_intr1.fy, infrared_intr1.ppy],
                          [0, 0, 1]])


print(infrared_intr)
print(infrared_intr1)

left_profile = cfg.get_stream(rs.stream.infrared, 1)
color_profile = cfg.get_stream(rs.stream.color)
extrinsics = left_profile.get_extrinsics_to(color_profile)
depth2color = np.eye(4)
depth2color[:3, :3] = np.reshape(extrinsics.rotation, [3, 3])
depth2color[:3, 3] = extrinsics.translation

print(depth2color)