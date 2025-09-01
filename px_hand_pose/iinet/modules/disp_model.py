import logging
import os.path

import timm
import torch
from modules.cost_volume import MsCostVolumeManager
from modules.networks import (CVEncoder, ResnetMatchingEncoder,UnetMatchingEncoder, DepthDecoderMSR)
import options
from torch import nn
import numpy as np

logger = logging.getLogger(__name__)

class DispModel(nn.Module):

    def __init__(self, opts):

        super().__init__()

        self.run_opts = opts

        self.num_ch_enc = [16, 24, 40, 112, 160]


        # iniitalize the first half of the U-Net, encoding the cost volume
        # and image prior image feautres
        if self.run_opts.cv_encoder_type == "multi_scale_encoder":

            self.cost_volume_net = CVEncoder(
                num_ch_cv=self.run_opts.max_disp // 2 ** (self.run_opts.matching_scale + 1),
                num_ch_encs=self.num_ch_enc,
                num_ch_outs=[24, 64, 128, 192, 256],
                lrcv_level=self.run_opts.matching_scale,
                multi_scale=self.run_opts.multiscale
            )

            dec_num_input_ch = (self.num_ch_enc[:self.run_opts.matching_scale - self.run_opts.multiscale]
                                + self.cost_volume_net.num_ch_enc)
        else:
            raise ValueError("Unrecognized option for cost volume encoder type!")

        # iniitalize the final depth decoder
        if self.run_opts.depth_decoder_name == "unet_pp":
            self.depth_decoder = DepthDecoderMSR(dec_num_input_ch,
                                               scales=self.run_opts.out_scale,
                                               lrcv_scale=self.run_opts.matching_scale + 1,
                                               multiscale=self.run_opts.multiscale)
        else:
            raise ValueError("Unrecognized option for depth decoder name!")

        if self.run_opts.feature_volume_type == "ms_cost_volume":
            cost_volume_class = MsCostVolumeManager
        else:
            raise ValueError("Unrecognized option for feature volume type {}!".format(self.run_opts.feature_volume_type))

        scale_factor = 2 ** (self.run_opts.matching_scale + 1)
        self.cost_volume = cost_volume_class(
            num_depth_bins=self.run_opts.max_disp // scale_factor,
            dot_dim=self.run_opts.dot_dim,
            disp_scale=self.run_opts.disp_scale // scale_factor,
            multiscale=self.run_opts.multiscale
        )

        # init the matching encoder. resnet is fast and is the default for
        # results in the paper, fpn is more accurate but much slower.
        if "resnet" == self.run_opts.matching_encoder_type:
            #prefp = opts.pre_weight_name
            prefp = None
            self.matching_model = ResnetMatchingEncoder(18,
                                                        self.run_opts.matching_feature_dims,
                                                        self.run_opts.matching_scale,
                                                        self.run_opts.multiscale,
                                                        pretrainedfp=prefp)
        elif "unet" == self.run_opts.matching_encoder_type:
            #prefp = opts.pre_weight_name
            prefp = None
            self.matching_model = UnetMatchingEncoder(self.run_opts.matching_feature_dims,
                                                      self.run_opts.matching_scale,
                                                      self.run_opts.multiscale,
                                                      pretrainedfp=prefp)
        else:
            raise ValueError("Unrecognized option for matching encoder type!")

    def forward(self, input:dict, only_uncer=False):

        left_image, right_image = input['left'], input['right']

        matching_left_feats, left_feats = self.matching_model(left_image)
        matching_right_feats, _ = self.matching_model(right_image)

        cost_volumes, hypos, priority = self.cost_volume(
                                                left_feats=matching_left_feats,
                                               right_feats=matching_right_feats,
                                           )

        if not only_uncer:
            filter_volumes = [cost_volume * confidence for cost_volume, confidence in zip(cost_volumes, priority['cconf'])]
            # Encode the cost volume and current image features
            cost_volume_features = self.cost_volume_net(
                                    filter_volumes,
                                    left_feats[self.run_opts.matching_scale - self.run_opts.multiscale:],
                                )
            left_feats = left_feats[:self.run_opts.matching_scale - self.run_opts.multiscale] + cost_volume_features

            # Decode into depth at multiple resolutions.
            depth_outputs = self.depth_decoder(left_feats, priority)
        else:
            depth_outputs = {}

        depth_outputs["coarse_disp"] = priority['cdisp'][0]
        depth_outputs["cost_volume"] = cost_volumes
        depth_outputs["hypos"] = hypos
        depth_outputs["confidence"] = priority['cconf'][0][0:1]

        return depth_outputs


if __name__ == '__main__':
    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    model = DispModel(opts)
