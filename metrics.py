import torch
import numpy as np
import torch.nn.functional as F

from nn_helpers.utils import type_tint, type_tfloat


def kl_divergence_np(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def bce_accuracy(pred_logits, targets, size_average=True):
    # https://github.com/jramapuram/helpers/blob/master/metrics.py
    cuda = pred_logits.is_cuda
    input_pred = F.sigmoid(pred_logits).type(type_tint(cuda))
    targets = targets.type(type_tint(cuda))
    reduction_fn = torch.mean if size_average else torch.sum
    return reduction_fn(input_pred.eq(targets).cpu().type(type_tfloat(False)))
