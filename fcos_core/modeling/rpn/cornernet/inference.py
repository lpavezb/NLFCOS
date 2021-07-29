import torch

from torch import nn
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.structures.boxlist_ops import boxlist_ml_nms
from fcos_core.structures.boxlist_ops import remove_small_boxes


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class CornerNetPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """

    def __init__(
            self,
            nms_thresh,
            fpn_post_nms_top_n,
            K=(60, 30, 15, 8, 4,),
            kernel=1,
            ae_threshold=1,
            num_dets=(1000, 500, 200, 50, 10),
            min_size=0
    ):
        """
        Arguments:
            K (list[int])
            kernel (int)
            ae_threshold (float)
            num_dets (list[int])
        """
        super(CornerNetPostProcessor, self).__init__()
        self.K = K
        self.kernel = kernel
        self.ae_threshold = ae_threshold
        self.num_dets = num_dets
        self.min_size = min_size
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

    def _nms(self, heat, kernel=1):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores, K=20):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

        topk_clses = (topk_inds / (height * width)).int()

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def forward_for_single_feature_map(self, tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, image_sizes, K, num_dets):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        kernel = self.kernel

        batch, cat, height, width = tl_heat.size()

        tl_heat = torch.sigmoid(tl_heat)
        br_heat = torch.sigmoid(br_heat)

        # perform nms on heatmaps
        tl_heat = self._nms(tl_heat, kernel=kernel)
        br_heat = self._nms(br_heat, kernel=kernel)

        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = self._topk(tl_heat, K=K)
        br_scores, br_inds, br_clses, br_ys, br_xs = self._topk(br_heat, K=K)

        tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
        tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
        br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
        br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

        if tl_regr is not None and br_regr is not None:
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            tl_regr = tl_regr.view(batch, K, 1, 2)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
            br_regr = br_regr.view(batch, 1, K, 2)

            tl_xs = tl_xs + tl_regr[..., 0]
            tl_ys = tl_ys + tl_regr[..., 1]
            br_xs = br_xs + br_regr[..., 0]
            br_ys = br_ys + br_regr[..., 1]

        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)
        tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
        tl_tag = tl_tag.view(batch, K, 1)
        br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
        br_tag = br_tag.view(batch, 1, K)
        dists = torch.abs(tl_tag - br_tag)
        tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
        br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
        scores = (tl_scores + br_scores) / 2

        # reject boxes based on classes
        tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
        br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
        cls_inds = (tl_clses != br_clses)

        # reject boxes based on distances
        dist_inds = (dists > self.ae_threshold)

        # reject boxes based on widths and heights
        width_inds = (br_xs < tl_xs)
        height_inds = (br_ys < tl_ys)

        scores[cls_inds] = -1
        scores[dist_inds] = -1
        scores[width_inds] = -1
        scores[height_inds] = -1

        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, num_dets)
        scores = scores.unsqueeze(2)

        bboxes = bboxes.view(batch, -1, 4)
        bboxes = _gather_feat(bboxes, inds)

        clses = tl_clses.contiguous().view(batch, -1, 1)
        clses = _gather_feat(clses, inds).float()

        tl_scores = tl_scores.contiguous().view(batch, -1, 1)
        tl_scores = _gather_feat(tl_scores, inds).float()
        br_scores = br_scores.contiguous().view(batch, -1, 1)
        br_scores = _gather_feat(br_scores, inds).float()
        clses = clses.view(batch, -1).int() + 1
        scores = scores.view(batch, -1)
        results = []
        h, w = image_sizes[0]
        for b in range(batch):
            boxlist = BoxList(bboxes[b], (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", clses[b])
            boxlist.add_field("scores", scores[b])
            boxlist.add_field("tl_scores", tl_scores[b])
            boxlist.add_field("br_scores", br_scores[b])
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, out_tl_heats, out_br_heats, out_tl_tags, out_br_tags, out_tl_regr, out_br_regr, image_sizes):
        """
        Arguments:
            out_tl_heats: list[Tensor]
            out_br_heats: list[Tensor]
            out_tl_tags: list[Tensor]
            out_br_tags: list[Tensor]
            out_tl_regr: list[Tensor]
            out_br_regr: list[Tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for level, (th, bh, tt, bt, tr, br) in enumerate(zip(out_tl_heats, out_br_heats, out_tl_tags, out_br_tags,
                                                         out_tl_regr, out_br_regr)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    th, bh, tt, bt, tr, br, image_sizes, self.K[level], self.num_dets[level]
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_cornernet_postprocessor(config):
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG

    box_selector = CornerNetPostProcessor(
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        K=[60, 30, 15, 8, 4],
        kernel=1,
        ae_threshold=1,
        num_dets=[1000, 500, 200, 50, 10]
    )

    return box_selector
