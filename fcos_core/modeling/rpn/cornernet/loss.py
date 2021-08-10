"""
This file contains specific functions for computing losses of CornerNet
file
"""
import os
import math
import torch
import numpy as np

from torch import nn
from .inference import _tranpose_and_gather_feat


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1, device="cpu"):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    y = torch.tensor(y, device=device)
    x = torch.tensor(x, device=device)
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6, device=heatmap.device)

    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


class CornerNetLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.pull_weight = 1e-1
        self.push_weight = 1e-1
        self.regr_weight = 1
        self.batch_size = self.batch_size = cfg.SOLVER.IMS_PER_BATCH // get_num_gpus()

        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.categories = num_classes

        stride = cfg.DATALOADER.SIZE_DIVISIBILITY
        img_shape = [cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN[0]]
        img_shape[0] = int(math.ceil(img_shape[0] / stride) * stride)
        img_shape[1] = int(math.ceil(img_shape[1] / stride) * stride)
        self.img_shape = img_shape

    def prepare_targets(self, targets, shape, device):
        max_tag_len = 128

        tl_heatmaps = torch.zeros((self.batch_size, self.categories, shape[0], shape[1]), dtype=torch.float32,
                                  device=device)
        br_heatmaps = torch.zeros((self.batch_size, self.categories, shape[0], shape[1]), dtype=torch.float32,
                                  device=device)
        tl_regrs = torch.zeros((self.batch_size, max_tag_len, 2), dtype=torch.float32, device=device)
        br_regrs = torch.zeros((self.batch_size, max_tag_len, 2), dtype=torch.float32, device=device)
        tl_tags = torch.zeros((self.batch_size, max_tag_len), dtype=torch.int64, device=device)
        br_tags = torch.zeros((self.batch_size, max_tag_len), dtype=torch.int64, device=device)
        tag_masks = torch.zeros((self.batch_size, max_tag_len), dtype=torch.bool, device=device)
        tag_lens = torch.zeros((self.batch_size,), dtype=torch.int32, device=device)

        for b_ind in range(self.batch_size):
            box_list = targets[b_ind]
            detections = box_list.bbox
            labels_per_detection = box_list.get_field("labels")
            for ind, detection in enumerate(detections):
                category = labels_per_detection[ind] - 1

                xtl, ytl = detection[0], detection[1]
                xbr, ybr = detection[2], detection[3]

                width_ratio = shape[1] / self.img_shape[1]
                height_ratio = shape[0] / self.img_shape[0]

                fxtl = (xtl * width_ratio)
                fytl = (ytl * height_ratio)
                fxbr = (xbr * width_ratio)
                fybr = (ybr * height_ratio)

                xtl = int(fxtl)
                ytl = int(fytl)
                xbr = int(fxbr)
                ybr = int(fybr)

                width = detection[2] - detection[0]
                height = detection[3] - detection[1]

                width = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)
                radius = gaussian_radius((height, width), 0.3)
                radius = max(0, int(radius))

                draw_gaussian(tl_heatmaps[b_ind, category], [xtl, ytl], radius)
                draw_gaussian(br_heatmaps[b_ind, category], [xbr, ybr], radius)

                tag_ind = tag_lens[b_ind]
                tl_regrs[b_ind, tag_ind, :] = torch.tensor([fxtl - xtl, fytl - ytl])
                br_regrs[b_ind, tag_ind, :] = torch.tensor([fxbr - xbr, fybr - ybr])
                tl_tags[b_ind, tag_ind] = ytl * shape[1] + xtl
                br_tags[b_ind, tag_ind] = ybr * shape[1] + xbr
                tag_lens[b_ind] += 1

        for b_ind in range(self.batch_size):
            tag_len = tag_lens[b_ind]
            tag_masks[b_ind, :tag_len] = True
        return tl_heatmaps, br_heatmaps, tl_regrs, br_regrs, tl_tags, br_tags, tag_masks

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

    def __call__(self, out, targets):
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

        # tl_heatmaps, br_heatmaps, tl_regrs, br_regrs, tl_tags, br_tags, tag_masks
        computed_targets = self.prepare_targets(targets, tl_heat.shape[2:], tl_heat.device)
        gt_tl_heat = computed_targets[0]
        gt_br_heat = computed_targets[1]
        gt_tl_regr = computed_targets[2]
        gt_br_regr = computed_targets[3]
        gt_tl_tags = computed_targets[4]
        gt_br_tags = computed_targets[5]
        gt_mask = computed_targets[6]

        tl_tag = _tranpose_and_gather_feat(tl_tag, gt_tl_tags)
        br_tag = _tranpose_and_gather_feat(br_tag, gt_br_tags)
        tl_regr = _tranpose_and_gather_feat(tl_regr, gt_tl_tags)
        br_regr = _tranpose_and_gather_feat(br_regr, gt_br_tags)

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
