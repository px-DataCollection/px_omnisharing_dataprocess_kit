# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *

if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--left_input_dir', default='none', type=str)
  parser.add_argument('--right_input_dir', default='none', type=str)
  parser.add_argument('--intrinsic_file', default='none', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--ckpt_dir', default='none', type=str, help='pretrained model path')
  parser.add_argument('--out_dir', default='none')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  os.makedirs(args.out_dir, exist_ok=True)

  ckpt_dir = args.ckpt_dir
  cfg = {}
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")

  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  model = FoundationStereo(cfg)

  ckpt = torch.load(ckpt_dir)
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model'])

  model.cuda()
  model.eval()

  left_input_dir = args.left_input_dir
  right_input_dir = args.right_input_dir

  for root,dirs,files in os.walk(left_input_dir):
    for file in files:
      if file.endswith('.png'):
        print(f"Processing {file}")
        left_file = os.path.join(left_input_dir, file)
        right_file = os.path.join(right_input_dir, file)
        code_dir = os.path.dirname(os.path.realpath(__file__))
        start_time = time.time()
        img0 = imageio.imread(left_file)
        if len(img0.shape) == 2:  # 灰度图
          img0 = np.stack([img0] * 3, axis=-1)  # 扩展为三通道
        img1 = imageio.imread(right_file)
        if len(img1.shape) == 2:  # 灰度图
          img1 = np.stack([img1] * 3, axis=-1)  # 扩展为三通道
        scale = args.scale
        assert scale<=1, "scale must be <=1"
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
        H,W = img0.shape[:2]
        img0_ori = img0.copy()
        logging.info(f"img0: {img0.shape}")

        img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
        img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.cuda.amp.autocast(True):
          if not args.hiera:
            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
          else:
            disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)
        print('inference time:', time.time()-start_time)
        
        if args.remove_invisible:
          yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
          us_right = xx-disp
          invalid = us_right<0
          disp[invalid] = np.inf

        if args.get_pc:
          with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
            baseline = float(lines[1])
          K[:2] *= scale
          depth = K[0,0]*baseline/disp
          # np.save(f'{args.out_dir}/depth_meter.npy', depth)
          depth_mm = depth * 1000 # convert to mm
          print('depth shape is', depth_mm.shape)
          output_file = os.path.join(args.out_dir, file)
          cv2.imwrite(output_file, depth_mm.astype(np.uint16))


