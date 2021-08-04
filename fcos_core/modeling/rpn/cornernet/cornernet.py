import math
import os
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn

from .inference import make_cornernet_postprocessor, _tranpose_and_gather_feat
from .loss import make_cornernet_loss_evaluator

INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        Convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )


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


class Convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(Convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class NonLocalBlock(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(NonLocalBlock, self).__init__()

        self.bottleneck_channels = in_channels // 2
        self.theta = nn.Conv2d(in_channels, self.bottleneck_channels, kernel_size=1, stride=1)
        self.phi = nn.Conv2d(in_channels, self.bottleneck_channels, kernel_size=1, stride=1)
        self.g = nn.Conv2d(in_channels, self.bottleneck_channels, kernel_size=1, stride=1)

        self.W = nn.Conv2d(self.bottleneck_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        batch = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]

        theta = self.theta(x)
        theta = torch.reshape(theta, (-1, self.bottleneck_channels))

        phi = self.phi(x)
        phi = torch.reshape(phi, (self.bottleneck_channels, -1))

        g = self.g(x)
        g = torch.reshape(g, (-1, self.bottleneck_channels))

        theta_phi = torch.matmul(theta, phi)
        theta_phi = F.softmax(theta_phi, dim=1)

        theta_phi_g = torch.matmul(theta_phi, g)
        theta_phi_g = torch.reshape(theta_phi_g, (batch, -1, height, width))

        w = self.W(theta_phi_g)
        z = w + x

        return z


class CornerNetModule(torch.nn.Module):
    """
    Module for CornerNet computation. Takes feature maps from the backbone and
    CornerNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(CornerNetModule, self).__init__()
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.categories = num_classes
        stride = cfg.DATALOADER.SIZE_DIVISIBILITY
        img_shape = [cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN[0]]
        img_shape[0] = int(math.ceil(img_shape[0] / stride) * stride)
        img_shape[1] = int(math.ceil(img_shape[1] / stride) * stride)
        self.img_shape = img_shape
        self.batch_size = cfg.SOLVER.IMS_PER_BATCH

        self.tl_nl = NonLocalBlock(cfg, 256)
        self.br_nl = NonLocalBlock(cfg, 256)
        # keypoint heatmaps
        self.tl_heat = make_kp_layer(in_channels, in_channels, num_classes)
        self.br_heat = make_kp_layer(in_channels, in_channels, num_classes)

        # tags
        self.tl_tags = make_kp_layer(in_channels, in_channels, 1)
        self.br_tags = make_kp_layer(in_channels, in_channels, 1)

        self.tl_heat[-1].bias.data.fill_(-2.19)
        self.br_heat[-1].bias.data.fill_(-2.19)

        self.tl_regr = make_kp_layer(in_channels, in_channels, 2)
        self.br_regr = make_kp_layer(in_channels, in_channels, 2)

        self.loss_evaluator = make_cornernet_loss_evaluator(cfg)
        self.box_selector = make_cornernet_postprocessor(cfg)

        # initialization
        for modules in [self.tl_nl, self.br_nl, self.tl_tags, self.br_tags, self.tl_regr, self.br_regr]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, images, features, targets=None):
        if self.training:
            return self.forward_train(features, targets)
        else:
            return self.forward_test(images.image_sizes, features)

    def compute_tensrors_per_level(self, targets, shape, device):
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
        return tl_heatmaps, br_heatmaps, tl_regrs, br_regrs, tl_tags, br_tags, tag_masks, tag_lens

    def forward_train(self, features, targets=None):
        feature = features[2]

        tl_heatmaps, br_heatmaps, tl_regrs, br_regrs, tl_tags, br_tags, tag_masks, tag_lens = \
            self.compute_tensrors_per_level(targets, feature.shape[2:], features[0].device)

        tl_nl = self.tl_nl(feature)
        br_nl = self.br_nl(feature)

        tl_heat, br_heat = self.tl_heat(tl_nl), self.br_heat(br_nl)
        tl_tag, br_tag = self.tl_tags(tl_nl), self.br_tags(br_nl)
        tl_regr, br_regr = self.tl_regr(tl_nl), self.br_regr(br_nl)

        tl_tag = _tranpose_and_gather_feat(tl_tag, tl_tags)
        br_tag = _tranpose_and_gather_feat(br_tag, br_tags)
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_tags)
        br_regr = _tranpose_and_gather_feat(br_regr, br_tags)

        out = [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]
        computed_targets = [tl_heatmaps, br_heatmaps, tl_regrs, br_regrs, tag_masks]
        return self.evaluate_loss(out, computed_targets)

    def forward_test(self, image_sizes, features):
        feature = features[2]

        tl_nl = self.tl_nl(feature)
        br_nl = self.br_nl(feature)

        tl_heat, br_heat = self.tl_heat(tl_nl), self.br_heat(br_nl)
        tl_tag, br_tag = self.tl_tags(tl_nl), self.br_tags(br_nl)
        tl_regr, br_regr = self.tl_regr(tl_nl), self.br_regr(br_nl)

        out = [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]
        return self.get_boxes(out, image_sizes)

    def evaluate_loss(self, out, computed_targets):

        focal_loss, pull_loss, push_loss, regr_loss = self.loss_evaluator(out, computed_targets)
        losses = {
            "focal_loss": focal_loss,
            "pull_loss": pull_loss,
            "push_loss": push_loss,
            "regr_loss": regr_loss
        }
        return None, losses

    def get_boxes(self, out, image_sizes):
        boxes = self.box_selector(out, image_sizes)
        return boxes, {}


def build_cornernet(cfg, in_channels):
    return CornerNetModule(cfg, in_channels)
