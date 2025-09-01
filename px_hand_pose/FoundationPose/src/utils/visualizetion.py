import numpy as np
import cv2


def prj_marker2img(point, R, T, K):
  point_cam = np.matmul(R, point) + T
  point_pix = np.matmul(K, point_cam)
  point_pix = point_pix / point_pix[2]
  return point_pix[:2].astype(int)


def draw_bbox(color_img,ob_in_cam,K):

  dims = [72, 88, 86]  # 649手环
  point_center = ob_in_cam[:3, 3] * 1000
  rot_mat = ob_in_cam[:3, :3]



  point_object = np.array([dims[0] / 2, dims[1] / 2, dims[2] / 2])
  point_img1 = prj_marker2img(point_object, rot_mat, point_center, K)

  point_object = np.array([-dims[0] / 2, dims[1] / 2, dims[2] / 2])
  point_img2 = prj_marker2img(point_object, rot_mat, point_center, K)

  point_object = np.array([dims[0] / 2, -dims[1] / 2, dims[2] / 2])
  point_img3 = prj_marker2img(point_object, rot_mat, point_center, K)

  point_object = np.array([-dims[0] / 2, -dims[1] / 2, dims[2] / 2])
  point_img4 = prj_marker2img(point_object, rot_mat, point_center, K)

  point_object = np.array([dims[0] / 2, dims[1] / 2, -dims[2] / 2])
  point_img5 = prj_marker2img(point_object, rot_mat, point_center, K)

  point_object = np.array([-dims[0] / 2, dims[1] / 2, -dims[2] / 2])
  point_img6 = prj_marker2img(point_object, rot_mat, point_center, K)

  point_object = np.array([dims[0] / 2, -dims[1] / 2, -dims[2] / 2])
  point_img7 = prj_marker2img(point_object, rot_mat, point_center, K)

  point_object = np.array([-dims[0] / 2, -dims[1] / 2, -dims[2] / 2])
  point_img8 = prj_marker2img(point_object, rot_mat, point_center, K)

  cv2.line(color_img, point_img5, point_img6, (255, 0, 0), 2)
  cv2.line(color_img, point_img6, point_img8, (255, 0, 0), 2)
  cv2.line(color_img, point_img8, point_img7, (255, 0, 0), 2)
  cv2.line(color_img, point_img5, point_img7, (255, 0, 0), 2)

  cv2.line(color_img, point_img1, point_img5, (0, 255, 0), 2)
  cv2.line(color_img, point_img2, point_img6, (0, 255, 0), 2)
  cv2.line(color_img, point_img3, point_img7, (0, 255, 0), 2)
  cv2.line(color_img, point_img4, point_img8, (0, 255, 0), 2)

  cv2.line(color_img, point_img1, point_img2, (0, 0, 255), 2)
  cv2.line(color_img, point_img2, point_img4, (0, 0, 255), 2)
  cv2.line(color_img, point_img4, point_img3, (0, 0, 255), 2)
  cv2.line(color_img, point_img1, point_img3, (0, 0, 255), 2)

  return color_img