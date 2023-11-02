from typing import Callable

import numpy as np
import torch
import torchvision


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, img_dir: str, annot_file: str, transform: Callable | None = None
    ):
        super().__init__(img_dir, annot_file)

        self.transform = transform

        self.classes = []
        self.coco_to_preds = {}
        self.preds_to_coco = {}
        for i, category_id in enumerate(sorted(self.coco.cats.keys())):
            self.classes.append(self.coco.cats[category_id]["name"])
            self.coco_to_preds[category_id] = i
            self.preds_to_coco[i] = category_id

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)

        img_id = self.ids[index]

        # Ignore multi-objects annotated by a single bounding box.
        target = [obj for obj in target if "iscrowd" not in obj or obj["iscrowd"] == 0]

        classes = torch.tensor(
            [self.coco_to_preds[obj["category_id"]] for obj in target],
            dtype=torch.int64,
        )
        boxes = torch.tensor([obj["bbox"] for obj in target], dtype=torch.float32)
        if boxes.shape[0] == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        width, height = img.size
        # [xmin, ymin, width, height] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]

        # Clip bounding box so that fits within the image area.
        boxes[:, ::2] = boxes[:, ::2].clamp(min=0, max=width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)

        target = {
            "image_id": torch.tensor(img_id, dtype=torch.int64),
            "classes": classes,
            "boxes": boxes,
            "img_size": torch.tensor((width, height), dtype=torch.int64),
            "orig_img_size": torch.tensor((width, height), dtype=torch.int64),
            "orig_img": torch.tensor(np.asanyarray(img)),
        }

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def to_coco_label(self, label: int):
        return self.preds_to_coco[label]
