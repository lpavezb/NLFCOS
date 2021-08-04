"""
This file contains specific functions for computing losses of CornerNet
file
"""
import math
import torch
from torch.nn import functional as F
from torch import nn
import os
from ..utils import concat_box_prediction_layers

from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist


INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class CornerNetLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.pull_weight = 1e-1
        self.push_weight = 1e-1
        self.regr_weight = 1

    def _sigmoid(self, x):
        x = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return x

    def ae_loss(self, tag0, tag1, mask):
        num = mask.sum(dim=1, keepdim=True).float()
        tag0 = tag0.squeeze()
        tag1 = tag1.squeeze()

        tag_mean = (tag0 + tag1) / 2

        tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
        tag0 = tag0[mask].sum()
        tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
        tag1 = tag1[mask].sum()
        pull = tag0 + tag1

        mask = mask.unsqueeze(1) + mask.unsqueeze(2)
        mask = mask.eq(2)
        num = num.unsqueeze(2)
        num2 = (num - 1) * num
        dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
        dist = 1 - torch.abs(dist)

        dist = nn.functional.relu(dist, inplace=True)
        dist = dist - 1 / (num + 1e-4)
        dist = dist / (num2 + 1e-4)
        dist = dist[mask]
        push = dist.sum()
        return pull, push

    def focal_loss(self, preds, gt):
        pos_inds = gt.eq(1)
        neg_inds = gt.lt(1)

        neg_weights = torch.pow(1 - gt[neg_inds], 4)

        loss = 0
        for pred in preds:
            pos_pred = pred[pos_inds]
            neg_pred = pred[neg_inds]

            pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
            neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

            num_pos = pos_inds.float().sum()
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()

            if pos_pred.nelement() == 0:
                loss = loss - neg_loss
            else:
                loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def regr_loss(self, regr, gt_regr, mask):
        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(gt_regr)

        regr = regr[mask]
        gt_regr = gt_regr[mask]

        regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
        regr_loss = regr_loss / (num + 1e-4)
        return regr_loss

    def __call__(self, out, computed_targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        focal_loss = 0
        pull_loss = 0
        push_loss = 0
        regr_loss = 0

        tl_heat = out[0]
        br_heat = out[1]
        tl_tag = out[2]
        br_tag = out[3]
        tl_regr = out[4]
        br_regr = out[5]

        gt_tl_heat = computed_targets[0]
        gt_br_heat = computed_targets[1]
        gt_tl_regr = computed_targets[2]
        gt_br_regr = computed_targets[3]
        gt_mask = computed_targets[4]

        tl_heats = self._sigmoid(tl_heat)
        br_heats = self._sigmoid(br_heat)

        focal_loss += self.focal_loss([tl_heats], gt_tl_heat)
        focal_loss += self.focal_loss([br_heats], gt_br_heat)

        pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
        pull_loss += self.pull_weight * pull
        push_loss += self.push_weight * push

        regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
        regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)

        regr_loss *= self.regr_weight

        return focal_loss, pull_loss, push_loss, regr_loss


def make_cornernet_loss_evaluator(cfg):
    loss_evaluator = CornerNetLossComputation(cfg)
    return loss_evaluator
