import random

import torch
from torch.utils.data import Dataset


def convert_to_xywh(boxes: torch.Tensor):
    """Convert from [xmin, ymin, xmax, ymax] to [x, y, width, height].

    Args:
        boxes (torch.Tensor): shape = [..., 4]
    """
    wh = boxes[..., 2:] - boxes[..., :2]
    xy = boxes[..., :2] + wh / 2
    boxes = torch.cat((xy, wh), dim=-1)
    return boxes


def convert_to_xyxy(boxes: torch.Tensor):
    """Convert from [x, y, width, height] to [xmin, ymin, xmax, ymax]

    Args:
        boxes (torch.Tensor): shape = [..., 4]
    """
    xymin = boxes[..., :2] - boxes[..., 2:] / 2
    xymax = boxes[..., 2:] + xymin
    boxes = torch.cat((xymin, xymax), dim=-1)
    return boxes


def calc_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """Calculate IoU.

    Args:
        boxes1: Bounding boxes, shape = [N, 4 (xmin, ymin, xmax, ymax)].
        boxes2: Bounding boxes, shape = [N, 4 (xmin, ymin, xmax, ymax)].
    """
    intersect_top_left = torch.maximum(boxes1[:, :2].unsqueeze(1), boxes2[:, :2])
    intersect_bottom_right = torch.minimum(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:])

    intersect_width_height = (intersect_bottom_right - intersect_top_left).clamp(min=0)
    intersect_areas = intersect_width_height.prod(dim=2)

    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_areas = areas1.unsqueeze(1) + areas2 - intersect_areas

    ious = intersect_areas / union_areas
    return ious, union_areas


def generate_subset(dataset: Dataset, ratio: float, random_seed: int = 0):
    size = int(len(dataset) * ratio)
    indices = list(range(len(dataset)))

    random.seed(random_seed)
    random.shuffle(indices)

    indices1, indices2 = indices[:size], indices[size:]
    return indices1, indices2

def collate_fn(batch):
    max_height = 0
    max_width = 0
    for img, _ in batch:
        h, w = img.shape[1:]
        max_height = max(max_height, h)
        max_width = max(max_width, w)

    height = (max_height + 31) // 32 * 32
    width = (max_width + 31) // 32 * 32

    imgs = batch[0][0].new_zeros((len(batch), 3, height, width))
    targets = []
    for i, (img, target) in enumerate(batch):
        h, w = img.shape[1:]
        imgs[i, :, :h, :w] = img
        targets.append(target)

    return imgs, targets
