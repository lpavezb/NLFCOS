import math

import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_cornernet_postprocessor, _tranpose_and_gather_feat
from .loss import make_cornernet_loss_evaluator

INF = 100000000


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
        self.use_bn = cfg.MODEL.FCOS.NON_LOCAL.USE_BN

        self.bottleneck_channels = in_channels // 2
        self.theta = nn.Conv2d(in_channels, self.bottleneck_channels, kernel_size=1, stride=1)
        self.phi = nn.Conv2d(in_channels, self.bottleneck_channels, kernel_size=1, stride=1)
        self.g = nn.Conv2d(in_channels, self.bottleneck_channels, kernel_size=1, stride=1)

        self.W = nn.Conv2d(self.bottleneck_channels, in_channels, kernel_size=1, stride=1)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(in_channels)

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
        if self.use_bn:
            w = self.bn(w)
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

    def forward(self, images, features, targets=None):
        if self.training:
            return self.forward_train(features, targets)
        else:
            return self.forward_test(images.image_sizes, features)

    def compute_tensrors_per_level(self, targets, shapes, device):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]

        targets_per_level = []
        for i, shape in enumerate(shapes):
            max_tag_len = max(shape)
            targets_per_level.append({
                "tl_heatmaps": torch.zeros((self.batch_size, self.categories, shape[0], shape[1]), dtype=torch.float32,
                                           device=device),
                "br_heatmaps": torch.zeros((self.batch_size, self.categories, shape[0], shape[1]), dtype=torch.float32,
                                           device=device),
                "tl_regrs": torch.zeros((self.batch_size, max_tag_len, 2), dtype=torch.float32, device=device),
                "br_regrs": torch.zeros((self.batch_size, max_tag_len, 2), dtype=torch.float32, device=device),
                "tl_tags": torch.zeros((self.batch_size, max_tag_len), dtype=torch.int64, device=device),
                "br_tags": torch.zeros((self.batch_size, max_tag_len), dtype=torch.int64, device=device),
                "tag_masks": torch.zeros((self.batch_size, max_tag_len), dtype=torch.bool, device=device),
                "tag_lens": torch.zeros((self.batch_size,), dtype=torch.int32, device=device)
            })
        for b_ind in range(self.batch_size):
            box_list = targets[b_ind]
            detections = box_list.bbox
            labels_per_detection = box_list.get_field("labels")
            for ind, detection in enumerate(detections):
                category = labels_per_detection[ind] - 1

                xtl, ytl = detection[0], detection[1]
                xbr, ybr = detection[2], detection[3]

                max_size = max((xbr - xtl), (ybr - ytl))
                level = 0
                for i, sizes in enumerate(object_sizes_of_interest):
                    if max_size < sizes[1]:
                        level = i
                        break
                level_tensors = targets_per_level[level]

                width_ratio = shapes[level][1] / self.img_shape[1]
                height_ratio = shapes[level][0] / self.img_shape[0]

                fxtl = (xtl * width_ratio)
                fytl = (ytl * height_ratio)
                fxbr = (xbr * width_ratio)
                fybr = (ybr * height_ratio)

                xtl = int(fxtl)
                ytl = int(fytl)
                xbr = int(fxbr)
                ybr = int(fybr)

                level_tensors["tl_heatmaps"][b_ind, category, ytl, xtl] = 1
                level_tensors["br_heatmaps"][b_ind, category, ybr, xbr] = 1
                tag_ind = level_tensors["tag_lens"][b_ind]
                level_tensors["tl_regrs"][b_ind, tag_ind, :] = torch.tensor([fxtl - xtl, fytl - ytl])
                level_tensors["br_regrs"][b_ind, tag_ind, :] = torch.tensor([fxbr - xbr, fybr - ybr])
                level_tensors["tl_tags"][b_ind, tag_ind] = ytl * shapes[level][1] + xtl
                level_tensors["br_tags"][b_ind, tag_ind] = ybr * shapes[level][1] + xbr
                level_tensors["tag_lens"][b_ind] += 1
        for b_ind in range(self.batch_size):
            for level in range(len(shapes)):
                level_tensors = targets_per_level[level]
                tag_len = level_tensors["tag_lens"][b_ind]
                level_tensors["tag_masks"][b_ind, :tag_len] = True
        return targets_per_level

    def forward_train(self, features, targets=None):

        out_tl_heats = []
        out_br_heats = []
        out_tl_tags = []
        out_br_tags = []
        out_tl_regr = []
        out_br_regr = []

        shapes = []
        stride = 8
        for i in range(len(features)):
            shapes.append([math.ceil(self.img_shape[0] / stride), math.ceil(self.img_shape[1] / stride)])
            stride *= 2
        targets_per_level = self.compute_tensrors_per_level(targets, shapes, features[0].device)
        for level, feature in enumerate(features):
            level_tensors = targets_per_level[level]
            tl_nl = self.tl_nl(feature)
            br_nl = self.br_nl(feature)
            tl_tags = level_tensors["tl_tags"]
            br_tags = level_tensors["br_tags"]

            tl_heat, br_heat = self.tl_heat(tl_nl), self.br_heat(br_nl)
            tl_tag, br_tag = self.tl_tags(tl_nl), self.br_tags(br_nl)
            tl_regr, br_regr = self.tl_regr(tl_nl), self.br_regr(br_nl)

            tl_tag = _tranpose_and_gather_feat(tl_tag, tl_tags)
            br_tag = _tranpose_and_gather_feat(br_tag, br_tags)
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_tags)
            br_regr = _tranpose_and_gather_feat(br_regr, br_tags)

            out_tl_heats.append(tl_heat)
            out_br_heats.append(br_heat)
            out_tl_tags.append(tl_tag)
            out_br_tags.append(br_tag)
            out_tl_regr.append(tl_regr)
            out_br_regr.append(br_regr)
        return self.evaluate_loss(out_tl_heats, out_br_heats, out_tl_tags,
                                  out_br_tags, out_tl_regr, out_br_regr, targets_per_level)

    def forward_test(self, image_sizes, features):
        out_tl_heats = []
        out_br_heats = []
        out_tl_tags = []
        out_br_tags = []
        out_tl_regr = []
        out_br_regr = []

        for feature in features:
            tl_nl = self.tl_nl(feature)
            br_nl = self.br_nl(feature)

            tl_heat, br_heat = self.tl_heat(tl_nl), self.br_heat(br_nl)
            tl_tag, br_tag = self.tl_tags(tl_nl), self.br_tags(br_nl)
            tl_regr, br_regr = self.tl_regr(tl_nl), self.br_regr(br_nl)

            out_tl_heats.append(tl_heat)
            out_br_heats.append(br_heat)
            out_tl_tags.append(tl_tag)
            out_br_tags.append(br_tag)
            out_tl_regr.append(tl_regr)
            out_br_regr.append(br_regr)
        return self.get_boxes(out_tl_heats, out_br_heats, out_tl_tags, out_br_tags,
                              out_tl_regr, out_br_regr, image_sizes)

    def evaluate_loss(self, out_tl_heats, out_br_heats, out_tl_tags,
                      out_br_tags, out_tl_regr, out_br_regr, targets):

        focal_loss, pull_loss, push_loss, regr_loss = self.loss_evaluator(out_tl_heats, out_br_heats, out_tl_tags,
                                                                          out_br_tags, out_tl_regr, out_br_regr,
                                                                          targets)
        losses = {
            "focal_loss": focal_loss,
            "pull_loss": pull_loss,
            "push_loss": push_loss,
            "regr_loss": regr_loss
        }
        return None, losses

    def get_boxes(self, out_tl_heats, out_br_heats, out_tl_tags, out_br_tags, out_tl_regr, out_br_regr, image_sizes):
        boxes = self.box_selector(out_tl_heats, out_br_heats, out_tl_tags, out_br_tags,
                                  out_tl_regr, out_br_regr, image_sizes)
        return boxes, {}


def build_cornernet(cfg, in_channels):
    return CornerNetModule(cfg, in_channels)
