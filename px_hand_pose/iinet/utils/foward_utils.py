import torch
import torch.nn.functional as F
from dataclasses import dataclass
from utils.summary_logger import TensorboardSummary
import torch.utils.data as data
from utils.utils import save_scalars, save_images, tensor2float, tensor2numpy, savepreprocess
import progressbar
import matplotlib.pyplot as plt
import os
import math
import sys
import cv2
import time

def gen_log_image(data: dict, outputs: dict, uncer_only=False):
    if not uncer_only:
        disp_gt = outputs['disp']
        valid_mask = outputs['validmask']
        disp_pred = (16 * outputs['disp_pred_s0'][0:1]).squeeze(1)
        coarse_disp = outputs['coarse_disp'][0:1].squeeze(1)
    else:
        disp_gt = outputs['displ']
        margin = torch.where(disp_gt % 2 < 1, disp_gt % 2,
                             disp_gt % 2 - 2)

        valid_mask = torch.logical_and(disp_gt > 0.0, disp_gt <= 192)
        disp_gt = disp_gt - margin
        disp_pred = (16 * outputs['coarse_disp'])[0:1].squeeze(1)
        coarse_disp = disp_pred

    confidence = outputs['confidence'][0:1].squeeze(1)
    px_err_map = (1.0 * (torch.abs(disp_pred - disp_gt) > 2)).masked_fill(~valid_mask, 0.0)
    epe_map = torch.abs(disp_pred - disp_gt).masked_fill(~valid_mask, 0.0).squeeze(1)

    image_outputs = {'disp_gt': disp_gt.squeeze(1),
                     'left_img': data['left'][0:1],
                     'right_img': data['right'][0:1],
                     'disp_est': disp_pred,
                     'epe_map': epe_map,
                     'coarse_disp': coarse_disp,
                     'confidence': confidence,
                     'px_error_map': px_err_map,
                     }
    return tensor2numpy(image_outputs)


def forward_pass(model, data, device, criterion, val=False, uncer_only=False):
    """
    forward pass of the model given input
    """

    inputs = {
        'left': data['left'].to(device),
        'right': data['right'].to(device),
        'disp_pyr': data['disp_pyr'].unsqueeze(1).to(device),
        'pos': data['pos'].to(device)
    }


    # forward pass
    outputs = model(inputs, uncer_only)

    # compute loss
    losses = criterion(inputs,  outputs, uncer_only)

    if not val:
        scalar_outputs = {'LRl1': losses['l1'][-1],
                          'HRl1': losses['l1'][0],
                          'grad': losses['grad'][0],
                          'normal': losses['normal'],
                          'focal': losses['focal'],
                          'epe': losses['epe'],
                          'd1': losses['d1'],
                          'px_error': losses['bad3']
                          }
    else:
        scalar_outputs = {'bad1': losses['bad1'],
                          'bad2': losses['bad2'],
                          'px_error': losses['bad3'],
                          'epe':losses['epe'],
                          'd1':losses['d1'],
                        }

    return losses, tensor2float(scalar_outputs), outputs


def train_one_epoch(model: torch.nn.Module,
                    data_loader: data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: torch.nn.Module,
                    device: torch.device,
                    epoch_idx: int,
                    summary: TensorboardSummary,
                    opts: dataclass,):
    """
    train model for 1 epoch
    """
    pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ",
                progressbar.Bar(), " ",
                progressbar.Timer(), ",", progressbar.ETA(), ",",
                progressbar.Variable('LRl1', width=1), ",", progressbar.Variable('HRl1', width=1), ",",
                progressbar.Variable('grad', width=1), ",", progressbar.Variable('normal', width=1), ",",progressbar.Variable('foc', width=1), ",",
                progressbar.Variable('epe', width=1), ",", progressbar.Variable('pxe', width=1),",", progressbar.Variable('d1', width=1)
                ]
    pbar = progressbar.ProgressBar(widgets=pwidgets,
                                   max_value=len(data_loader),
                                    prefix="Epoch {}/{}: ".format(epoch_idx, opts.epochs)).start()

    model.train()
    criterion.train()
    global_step = len(data_loader) * epoch_idx
    summary.avgMeter.clear()
    uncer_only = True if epoch_idx < 13 else False
    #uncer_only = False

    if opts.dataset != 'sceneflow':
        uncer_only = False

    for batch_idx, data in enumerate(data_loader):

        losses, scalar_outputs, outputs = forward_pass(model, data, device, criterion, uncer_only=uncer_only)

        # terminate training if exploded
        if not math.isfinite(losses['aggregated'].item()):
            print("Loss is {}, stopping training".format(losses['aggregated'].item()))
            sys.exit(1)

        optimizer.zero_grad()
        losses['aggregated'].backward()


        optimizer.step()
        summary.avgMeter.update(scalar_outputs)
        if batch_idx >= len(data_loader) - 1:
            save_scalars(summary.writer, 'fulltrain', summary.avgMeter.avg_data, global_step)
        if (epoch_idx * len(data_loader) + batch_idx) % opts.summary_freq == 0:

            image_outputs = gen_log_image(data, outputs, uncer_only)

            save_scalars(summary.writer, 'train', scalar_outputs, global_step + batch_idx)
            save_images(summary.writer, 'train', image_outputs, global_step+batch_idx)

            del image_outputs

        pbar.update(batch_idx,
                    LRl1="{:.3f}|{:.3f}".format(scalar_outputs["LRl1"],
                                              summary.avgMeter.avg_data["LRl1"]),
                    HRl1="{:.3f}|{:.3f}".format(scalar_outputs["HRl1"],
                                               summary.avgMeter.avg_data["HRl1"]),
                    grad="{:.3f}|{:.3f}".format(scalar_outputs["grad"],
                                               summary.avgMeter.avg_data["grad"]),
                    normal="{:.3f}|{:.3f}".format(scalar_outputs["normal"],
                                               summary.avgMeter.avg_data["normal"]),
                    foc="{:.3f}|{:.3f}".format(scalar_outputs["focal"],
                                                  summary.avgMeter.avg_data["focal"]),
                    epe="{:.3f}|{:.3f}".format(scalar_outputs["epe"],
                                                summary.avgMeter.avg_data["epe"]),
                    pxe="{:.3f}|{:.3f}".format(scalar_outputs["px_error"],
                                               summary.avgMeter.avg_data["px_error"]),
                    d1="{:.3f}|{:.3f}".format(scalar_outputs["d1"],
                                                 summary.avgMeter.avg_data["d1"])
                    )
        del scalar_outputs, outputs, losses, data
        # clear cache
    summary.output_json('train_epoch_logger.json', epoch_idx)
    pbar.finish()
    torch.cuda.empty_cache()
    return


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: data.DataLoader,
             device: torch.device,
             epoch_idx: int,
             opts: dataclass,
             summary: TensorboardSummary):

    pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ",
                progressbar.Bar(), " ",
                progressbar.Timer(), ",", progressbar.ETA(), ",",
                progressbar.Variable('bad1', width=1), ",", progressbar.Variable('bad2', width=1), ",",
                progressbar.Variable('epe', width=1), ",", progressbar.Variable('bad3', width=1),",", progressbar.Variable('d1', width=1)
                ]
    pbar = progressbar.ProgressBar(widgets=pwidgets,
                                   max_value=len(data_loader),
                                   prefix="Epoch {}/{}: ".format(epoch_idx, opts.epochs)).start()
    model.eval()
    criterion.eval()
    summary.avgMeter.clear()

    global_step = epoch_idx * len(data_loader)
    uncer_only = True if epoch_idx < 13 else False
    if opts.dataset != 'sceneflow':
        uncer_only = False

    for batch_idx, data in enumerate(data_loader):
        # forward pass

        losses, scalar_outputs, outputs = forward_pass(model, data, device, criterion, val=True, uncer_only=uncer_only)

        summary.avgMeter.update(scalar_outputs)
        if (epoch_idx * len(data_loader) + batch_idx) % (opts.summary_freq_eval) == 0:

            image_outputs = gen_log_image(data, outputs, uncer_only)
            save_scalars(summary.writer, 'val', scalar_outputs, global_step + batch_idx)
            save_images(summary.writer, 'val', image_outputs, global_step + batch_idx)


            if opts.save_eval:
                save_path = os.path.join(summary.directory, 'eval_pics')
                os.makedirs(save_path, exist_ok=True)
                #fn = (data['left_filename'][0].split('/')[-2]+(data['left_filename'][0].split('/')[-1]).split('.')[0])
                fn = batch_idx
                plt.imsave(os.path.join(save_path, '%s_dispCoarse.jpg' % fn),
                           savepreprocess(image_outputs['coarse_disp']) / 255.)
                plt.imsave(os.path.join(save_path, '%s_dispEst.jpg' % fn), savepreprocess(image_outputs['disp_est']) / 255.)
                plt.imsave(os.path.join(save_path, '%s_dispGT.jpg' % fn), savepreprocess(image_outputs['disp_gt']) / 255.)
                plt.imsave(os.path.join(save_path, '%s_conf.png' % fn),savepreprocess(image_outputs['confidence']))
                cv2.imwrite(os.path.join(save_path, '%s_PXE.png' % fn), (image_outputs['px_error_map'][0] * 255).transpose(1, 2, 0))
                plt.imsave(os.path.join(save_path, '%s_left.png' % fn), savepreprocess(image_outputs['left_img'], True))
                plt.imsave(os.path.join(save_path, '%s_right.png' % fn), savepreprocess(image_outputs['right_img'], True))
            del image_outputs

        pbar.update(batch_idx,
                    bad1="{:.3f}|{:.3f}".format(scalar_outputs["bad1"],
                                                 summary.avgMeter.avg_data["bad1"]),
                    bad2="{:.3f}|{:.3f}".format(scalar_outputs["bad2"],
                                                 summary.avgMeter.avg_data["bad2"]),
                    bad3="{:.3f}|{:.3f}".format(scalar_outputs["px_error"],
                                                summary.avgMeter.avg_data["px_error"]),
                    epe="{:.3f}|{:.3f}".format(scalar_outputs["epe"],
                                               summary.avgMeter.avg_data["epe"]),
                    d1="{:.3f}|{:.3f}".format(scalar_outputs["d1"],
                                              summary.avgMeter.avg_data["d1"])
                    )

        del scalar_outputs, outputs, losses

    save_scalars(summary.writer, 'fullval', summary.avgMeter.avg_data, epoch_idx)
    summary.output_json('eval_epoch_logger.json', epoch_idx)
    pbar.finish()
    torch.cuda.empty_cache()

    return