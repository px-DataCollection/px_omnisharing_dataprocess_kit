import os
import time
from datetime import datetime

import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
from torchvision.transforms import transforms

from align_image import align_image_by_depth
from init_logger import get_logger
from pcd import depth2pcd


class IINetTRT:
    def __init__(self,
                 engine_path,
                 ratio=1.0,
                 save_ply=False,
                 log_lvl='INFO',
                 vis_dir: str = None,
                 **kwargs):
        '''
        Initialize the llNet model for depth estimation.
        :param ratio: 深度缩放比例，默认1.0
        '''
        self.log_lvl = log_lvl
        self.logger = get_logger(name='IINetTRT', log_lvl=log_lvl, **kwargs)
        self.vis_dir = vis_dir
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

        self.save_ply = save_ply

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for i, binding in enumerate(self.engine):
            shape = self.engine.get_tensor_shape(binding)
            size = trt.volume(self.engine.get_tensor_shape(binding))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))

            # 分配主机和设备内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if i == 0 or i == 1:
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

        # self.IINet_model = TRTModule()
        # self.IINet_model.load_state_dict(torch.load(engine_path, map_location=self.device))
        # self.IINet_model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.ratio = ratio

    def infer(self, left_img, right_img):
        # 支持PyTorch Tensor输入（需在GPU上）
        if isinstance(left_img, torch.Tensor):
            # 获取Tensor的底层指针
            left_ptr = left_img.contiguous().data_ptr()
            right_ptr = right_img.contiguous().data_ptr()

            # 直接拷贝GPU内存
            cuda.memcpy_dtod(self.inputs[0]['device'], left_ptr, left_img.nbytes)
            cuda.memcpy_dtod(self.inputs[1]['device'], right_ptr, right_img.nbytes)
        else:
            # 原始Numpy数组处理逻辑
            np.copyto(self.inputs[0]['host'], left_img.ravel())
            np.copyto(self.inputs[1]['host'], right_img.ravel())

            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            cuda.memcpy_htod_async(self.inputs[1]['device'], self.inputs[1]['host'], self.stream)

        # 执行推理（TensorRT 10.x兼容写法）
        self.context.execute_v2(bindings=self.bindings)

        # 获取结果
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()
        return [out['host'] for out in self.outputs]

    def predict(self, left_img, right_img, K_color, K_depth, baseline, color2depth, scale=1, fill_zero=True, align=True,
                rgb=None, dt_str=None):
        preprocess_t = time.time()
        if dt_str is None:
            dt_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

        # 如果是单通道的图像，转换为三通道
        if len(left_img.shape) == 2:
            left_img = np.stack([left_img] * 3, axis=-1)
        if len(right_img.shape) == 2:
            right_img = np.stack([right_img] * 3, axis=-1)

        h, w, c = left_img.shape
        top_pad = (32 - (h % 32)) % 32
        right_pad = (32 - (w % 32)) % 32

        if not (top_pad == 0 and right_pad == 0):
            left_img = np.lib.pad(left_img, ((top_pad, 0), (0, right_pad), (0, 0)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((top_pad, 0), (0, right_pad), (0, 0)), mode='constant',
                                   constant_values=0)
        print("padding_time: ", time.time() - preprocess_t, "s")

        # left_img = left_img.astype(np.float32)
        # left_img = (left_img / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        # left_img = np.transpose(left_img, (2, 0, 1))
        # left_img = np.expand_dims(left_img, axis=0)
        #
        # right_img = right_img.astype(np.float32)
        # right_img = (right_img / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        # right_img = np.transpose(right_img, (2, 0, 1))
        # right_img = np.expand_dims(right_img, axis=0)

        transform_t = time.time()

        left_img = self.transform(left_img).unsqueeze(0).to(self.device)
        right_img = self.transform(right_img).unsqueeze(0).to(self.device)

        torch.cuda.synchronize()
        print("transform_time: ", time.time() - transform_t, "s")
        print("preprocess_time: ", time.time() - preprocess_t, "s")

        # forward pass
        forward_t = time.time()
        output = self.infer(left_img, right_img)[0].reshape(left_img.shape[2], left_img.shape[3])
        print("forward_time: ", time.time() - forward_t, "s")

        postprocess_t = time.time()
        if right_pad != 0:
            disp_pred_np = (16 * output)[top_pad:, :-right_pad]  # 1,h,w
        else:
            disp_pred_np = (16 * output)[top_pad:, :]  # 1,h,w

        focal_length = K_depth[0, 0]
        depth = focal_length * baseline / (disp_pred_np + 1e-6)
        depth = depth * self.ratio

        # Align depth to color
        torch.cuda.synchronize()
        aligned_t = time.time()
        if align:
            depth = (align_image_by_depth(depth, K_color, K_depth, color2depth, self.device, rgb, to_depth=False,
                                          fill_zero=fill_zero) * 1000).astype(np.uint16)
        else:
            depth = (depth * 1000).cpu().numpy().astype(np.uint16)
        torch.cuda.synchronize()
        print("align_time: ", time.time() - aligned_t, "s")

        if scale and scale != 1:
            depth = depth * scale

        if self.save_ply:
            depth2pcd(depth, K_color, os.path.join(self.vis_dir, f"{dt_str}_depth.ply"), rgb=rgb)
        print("postprocess_time: ", time.time() - postprocess_t, "s")
        return depth

    def stop(self):
        del self.context
        del self.engine