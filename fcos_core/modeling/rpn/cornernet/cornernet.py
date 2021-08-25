import math
import os
import torch
import torch.nn.functional as F

from torch import nn

from .inference import make_cornernet_postprocessor
from .loss import make_cornernet_loss_evaluator


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        Convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )


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
        theta = torch.reshape(theta, (batch, -1, self.bottleneck_channels))

        phi = self.phi(x)
        phi = torch.reshape(phi, (batch, self.bottleneck_channels, -1))

        g = self.g(x)
        g = torch.reshape(g, (batch, -1, self.bottleneck_channels))

        theta_phi = torch.bmm(theta, phi)
        theta_phi = F.softmax(theta_phi, dim=1)

        theta_phi_g = torch.bmm(theta_phi, g)
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
        self.batch_size = cfg.SOLVER.IMS_PER_BATCH // get_num_gpus()

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
        outs = []
        shapes = []

        shapes.append(features.shape[2:])
        tl_nl = self.tl_nl(features)
        br_nl = self.br_nl(features)

        tl_heat, br_heat = self.tl_heat(tl_nl), self.br_heat(br_nl)
        tl_tag, br_tag = self.tl_tags(tl_nl), self.br_tags(br_nl)
        tl_regr, br_regr = self.tl_regr(tl_nl), self.br_regr(br_nl)
        outs.append([tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr])
        if self.training:
            return self.evaluate_loss(outs, targets, shapes)
        else:
            return self.get_boxes(outs, images.image_sizes)

    def evaluate_loss(self, out, computed_targets, shapes):

        focal_loss, pull_loss, push_loss, regr_loss = self.loss_evaluator(out, computed_targets, shapes)
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
