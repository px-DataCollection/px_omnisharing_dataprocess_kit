import cv2


realsense_depth = cv2.imread('none', cv2.IMREAD_ANYDEPTH)
stereo_depth = cv2.imread('none', cv2.IMREAD_ANYDEPTH)

print(realsense_depth.shape)