import cv2
import os
import numpy as np
from yolo import yolo
import argparse


# ###
# obj_dict = {
#     # 'fenda': (19, 0),
#     # 'aobao': (130, 1),
#     # 'libai': (244, 2),
#     # 'leshi': (328, 3)
#
#     # 'Coconut_Juice': (1, 0),
#     # 'canned_Pepsi': (16, 1),
#     # 'shuang_wai_wai': (30, 2),
#     # 'Wang_Laoji': (31, 3),
#     # 'gear': (51, 4),
#     # 'lactic_acid_bacteria': (84, 5),
#     # 'two_pin_plug': (111, 6),
#     # 'small_watering_can': (121, 7),
#     # 'dabao': (139, 8),
#     # 'weimengxiansheng': (175, 9),
#     # 'liangkongmutou': (191, 10),
#     # 'blue_thermos_cup': (237, 11),
#     # 'libai': (244, 12),
#     # 'guashuiqi': (292, 13),
#     # 'ganmianzhang': (294, 14),
#     # 'refrigerator_deodorizer': (296, 15),
#     # 'milk_teapot': (331, 16),
#     # 'yaomishao': (343, 17),
#     # 'yashuabei375': (375, 18),
#     # 'paotengpian': (410, 19),
#     # 'dengpao437': (437, 20)
#
#     # 'finger524' : (524, 0)
#
#     'bracelet642': (642, 0),
#     'bracelet643': (643, 1)
#
#     # 'shuangwaiwai': (30, 0),
#     # 'yezishui': (100001, 1),
#     # 'chuangkoutiehe': (100002, 2)
#
#
#
#     }
#
# obj_id = 524
# symmetry_obj = []
# obj_name = ['bracelet642','bracelet643']
# # obj_name = ['Coconut_Juice','canned_Pepsi','shuang_wai_wai','Wang_Laoji','gear','lactic_acid_bacteria','two_pin_plug','small_watering_can','dabao','weimengxiansheng','liangkongmutou','blue_thermos_cup','libai',
# #             'guashuiqi','ganmianzhang','refrigerator_deodorizer','milk_teapot','yaomishao','yashuabei375','paotengpian','dengpao437']
# # obj_name = ['shuangwaiwai','yezishui','chuangkoutiehe']
# target_ids = []
# yolo_target_ids = []
# id_map = {}
# symmetry_map = {}
# count = 0
# for obj in obj_name:
#     target_ids.append(obj_dict[obj][0])
#     yolo_target_ids.append((obj_dict[obj][1]))
#     id_map[obj_dict[obj][0]] = count
#     count += 1
#     if obj in symmetry_obj:
#         symmetry_map[obj_dict[obj][0]] = True
#     else:
#         symmetry_map[obj_dict[obj][0]] = False

# def get_bbox_masks_from_yolo(results, target_ids, yolo_target_ids):
#     bboxes = []
#     masks = []
#     detected_ids = []
#     for result in results:
#         # if result[0].item() in yolo_target_ids:
#         if result[0].item() in [1]:
#             bboxes.append(result[1])
#             model_ids = target_ids[yolo_target_ids.index(result[0].item())]
#             detected_ids.append(model_ids)
#             aug_ratio = model_ids / 1000
#             # masks.append(result[2] * 0.)
#             masks.append(result[2]*aug_ratio) # 这里需要根据训练的时候设定值相应的修改
#             # if model_ids == 19:
#             #     masks.append(result[2] * 0.1)
#             # elif model_ids == 130:
#             #     masks.append(result[2] * 0.2)
#             # elif model_ids == 244:
#             #     masks.append(result[2] * 0.3)
#             # elif model_ids == 328:
#             #     masks.append(result[2] * 0.4)
#     return bboxes, masks, detected_ids

# yolo = yolo()

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
    parser.add_argument('--source_dir', default=f'{code_dir}/../assets/left.png', type=str)
    parser.add_argument('--model_path', default='./yolo.pt', type=str)
    parser.add_argument('--obj_name_list', default=['obj'], type=str)
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

    # yolo = yolo()
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
    # mask[120:600, 320:960] = masks[i]
    tmp_mask[8:712, 160:1120] = masks[0]

    if obj_name_list == "['bracelet_649']":
        mask_save_dir = os.path.join(source_dir, 'right_bracelet_masks')
    elif obj_name_list == "['bracelet_648']":
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

    # for i in range(len(masks)):
    #     mask_filename = os.path.join(mask_save_dir, "1725507976895.png")
    #     mask = masks[i].copy()  # 暂时只支持图片中只含有单个检测物体
    #     mask[mask != 0] = 255
    #     mask.astype(np.uint8)
    #     print('len mask shape',len(mask.shape))
    #     cv2.imwrite(mask_filename, mask)

    mask_filename = os.path.join(mask_save_dir, "1.png")
    print('len masks', len(masks))
    mask_final[tmp_mask != 0] = 255
    # for i in range(len(masks)):
    #
    #     mask = masks[i].copy()  # 暂时只支持图片中只含有单个检测物体
    #     mask_final[mask != 0] = 255
    #     # mask.astype(np.uint8)

    print('len mask shape', len(mask_final.shape))
    # cv2.imshow('mask',mask_final)
    # cv2.waitKey(0)
    print('mask_filename', mask_filename)
    cv2.imwrite(mask_filename, mask_final)