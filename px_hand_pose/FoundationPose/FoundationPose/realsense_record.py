## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
import os
from datetime import datetime

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 15)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Get the absolute path to the subfolder
# script_dir = os.path.dirname(os.path.abspath(__file__))

script_dir = ('none')
subfolder_depth = os.path.join(script_dir, "depth")
subfolder_rgb = os.path.join(script_dir, "rgb")
subfolder_depth_unaligned = os.path.join(script_dir, "depth_unaligned")
subfolder_rgb_unaligned = os.path.join(script_dir, "rgb_unaligned")

# Check if the subfolder exists, and create it if it does not
if not os.path.exists(subfolder_depth):
    os.makedirs(subfolder_depth)
if not os.path.exists(subfolder_rgb):
    os.makedirs(subfolder_rgb)
if not os.path.exists(subfolder_depth_unaligned):
    os.makedirs(subfolder_depth_unaligned)
if not os.path.exists(subfolder_rgb_unaligned):
    os.makedirs(subfolder_rgb_unaligned)

# Create all

RecordStream = False

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        unaligned_depth_frame = frames.get_depth_frame()
        unaligned_color_frame = frames.get_color_frame()

        # Get instrinsics from aligned_depth_frame
        intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image)
        )  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where(
            (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
            grey_color,
            color_image,
        )

        unaligned_depth_image = np.asanyarray(unaligned_depth_frame.get_data())
        unaligned_rgb_image = np.asanyarray(unaligned_color_frame.get_data())

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
        cv2.imshow("Align Example", images)

        key = cv2.waitKey(1)

        # Start saving the frames if space is pressed once until it is pressed again
        if key & 0xFF == ord(" "):
            if not RecordStream:
                time.sleep(0.2)
                RecordStream = True

                with open(os.path.join(script_dir, "cam_K.txt"), "w") as f:
                    f.write(f"{intrinsics.fx} {0.0} {intrinsics.ppx}\n")
                    f.write(f"{0.0} {intrinsics.fy} {intrinsics.ppy}\n")
                    f.write(f"{0.0} {0.0} {1.0}\n")

                print("Recording started")
            else:
                RecordStream = False
                print("Recording stopped")

        if RecordStream:
            framename = int(round(time.time() * 1000))
            # framename = datetime.now().strftime('%Y:%m:%d:%H:%M:%S:%f')

            # Define the path to the image file within the subfolder
            image_path_depth = os.path.join(subfolder_depth, f"{framename}.png")
            image_path_rgb = os.path.join(subfolder_rgb, f"{framename}.png")
            image_path_depth_unaligned = os.path.join(subfolder_depth_unaligned, f"{framename}.png")
            image_path_rgb_unaligned = os.path.join(subfolder_rgb_unaligned, f"{framename}.png")

            cv2.imwrite(image_path_depth, depth_image)
            cv2.imwrite(image_path_rgb, color_image)
            cv2.imwrite(image_path_depth_unaligned, unaligned_depth_image)
            cv2.imwrite(image_path_rgb_unaligned, unaligned_rgb_image)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord("q") or key == 27:

            cv2.destroyAllWindows()

            break
finally:
    pipeline.stop()