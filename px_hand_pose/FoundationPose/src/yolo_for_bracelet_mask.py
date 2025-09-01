import cv2
import os
import numpy as np
import sys

import argparse

src_path = os.path.join(os.path.dirname(__file__), "..")
if src_path not in sys.path:
    sys.path.append(src_path)

from yolo import yolo

def get_bbox_masks_from_yolo(results, obj_name_list):
    bboxes = []
    masks = []
    detected_ids = []
    for result in results:
        # if result[0].item() in yolo_target_ids:
        if result['name'] in obj_name_list:
            bboxes.append(result['bbox_cxywh'])
            aug_ratio = 1
            masks.append(result['mask']*aug_ratio) # 这里需要根据训练的时候设定值相应的修改
    return bboxes, masks, detected_ids

if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default=f'./RGBD_0', type=str)
    parser.add_argument('--model_path', default='./detect_bracelet_648_649_v_0.0.1.pt', type=str)
    parser.add_argument('--obj_name_list',
                        nargs='+',  # 接受一个或多个参数
                        default=['bracelet_649'],
                        help='List of object names')
    args = parser.parse_args()
    source_dir = args.source_dir
    model_path = args.model_path
    obj_name_list = args.obj_name_list

    yolo = yolo(model_path)

    color_file = os.path.join(source_dir, 'color', '1.png')
    color_img = cv2.imread(color_file)
    ##########进行yolo检测

    #############将图片裁切成 960*704
    # 计算裁剪区域
    start_x, end_x = 160, 1120
    start_y, end_y = 8, 712
    # 执行裁剪（注意行列顺序：高度在前，宽度在后）
    cropped_img = color_img[start_y:end_y, start_x:end_x]
    yolo_results = yolo.predict(cropped_img, conf=0.5)
    bboxes, masks, detected_ids = get_bbox_masks_from_yolo(yolo_results, obj_name_list)

    # print('bboxes',bboxes)

    if (len(bboxes) == 0):
        print('No target detected.')
        exit(0)
    # 只支持一个mask
    bboxes[0][0] += start_x
    bboxes[0][1] += start_y

    tmp_mask = np.zeros((720, 1280), dtype=np.float32)
    tmp_mask[8:712, 160:1120] = masks[0]

    if obj_name_list == ['bracelet_649']:
        mask_save_dir = os.path.join(source_dir, 'right_bracelet_masks')
    elif obj_name_list == ['bracelet_648']:
        mask_save_dir = os.path.join(source_dir, 'left_bracelet_masks')
    else:
        print('obj_name_list not supported')
        exit(0)
    # 检查路径是否存在
    if not os.path.exists(mask_save_dir):
        print(f"Creating directory: {mask_save_dir}")
        # 如果路径不存在，则创建它
        os.makedirs(mask_save_dir)

    mask_final = np.zeros((720, 1280), dtype=np.uint8)

    mask_filename = os.path.join(mask_save_dir, "1.png")
    print('len masks', len(masks))
    mask_final[tmp_mask != 0] = 255
    print('len mask shape', len(mask_final.shape))
    print('mask_filename', mask_filename)
    cv2.imwrite(mask_filename, mask_final)