#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import copy

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

_device = torch.device('cuda:0')





