from packaging import version
from collections import namedtuple
from inspect import getargspec

import numpy as np

from sklearn.metrics import roc_auc_score

import torch
from torch.nn.attention import SDPBackend

def set_device_config(flash):
    assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

    # determine efficient attention configs for cuda and cpu
    cpu_config = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
    if torch.cuda.is_available() and flash:
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # A100 GPU detected, using flash attention if input tensor is on cuda
            cuda_config = [SDPBackend.FLASH_ATTENTION]
        else:
            # Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
            cuda_config = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
    else:
        cuda_config = []

    return cpu_config, cuda_config

def get_args(values):
    args = {}
    for i in getargspec(values['self'].__init__).args[1:]:
        args[i] = values[i]
    return args

def auc_metric(target, proba, multi_class='ovo'):
    if len(np.unique(target)) > 2:
        return roc_auc_score(target, proba, multi_class=multi_class)
    else:
        return roc_auc_score(target, proba[:, 1])
