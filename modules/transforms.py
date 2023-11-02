import random
from typing import Callable, Sequence

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, img: Image.Image, target: dict):
        if random.random() < self.prob:
            img = F.hflip(img)

            width = img.size[0]
            target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return img, target


class RandomSizeCrop:
    def __init__(self, scale: Sequence[float], ratio: Sequence[float]):
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img: Image.Image, target: dict):
        width, height = img.size

        top, left, cropped_height, cropped_width = T.RandomResizedCrop.get_params(
            img, self.scale, self.ratio
        )
        img = F.crop(img, top, left, cropped_height, cropped_width)

        target["boxes"][:, ::2] -= left
        target["boxes"][:, 1::2] -= top

        target["boxes"][:, ::2] = target["boxes"][:, ::2].clamp(
            min=0, max=cropped_width
        )
        target["boxes"][:, 1::2] = target["boxes"][:, 1::2].clamp(
            min=0, max=cropped_height
        )

        # Keep bounding boxes which the area > 0.
        keep = (target["boxes"][:, 2] > target["boxes"][:, 0]) & (
            target["boxes"][:, 3] > target["boxes"][:, 1]
        )
        target["classes"] = target["classes"][keep]
        target["boxes"] = target["boxes"][keep]

        target["img_size"] = torch.tensor(
            (cropped_width, cropped_height), dtype=torch.int64
        )
        return img, target


class RandomResize:
    def __init__(self, min_sizes: Sequence[int], max_size: int):
        self.min_sizes = min_sizes
        self.max_size = max_size

    def _get_target_size(self, min_side: int, max_side: int, target: int):
        max_side = int(max_side * target / min_side)
        min_side = target

        if max_side > self.max_size:
            min_side = int(min_side * self.max_size / max_side)
            max_side = self.max_size
        return min_side, max_side

    def __call__(self, img: Image.Image, target: dict):
        min_size = random.choice(self.min_sizes)

        width, height = img.size

        if width < height:
            resized_width, resized_height = self._get_target_size(
                width, height, min_size
            )
        else:
            resized_height, resized_width = self._get_target_size(
                height, width, min_size
            )

        img = F.resize(img, (resized_height, resized_width))

        ratio = resized_width / width
        target["boxes"] *= ratio
        target["img_size"] = torch.tensor(
            (resized_width, resized_height), dtype=torch.int64
        )
        return img, target


class ToTensor:
    def __call__(self, img: Image.Image, target: dict):
        img = F.to_tensor(img)
        return img, target


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor, target: dict):
        img = F.normalize(img, self.mean, self.std)
        return img, target


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, img: Image.Image, target: dict):
        for transform in self.transforms:
            img, target = transform(img, target)
        return img, target


class RandomSelect:
    def __init__(self, transform1: Callable, transform2: Callable, prob: float = 0.5):
        self.transform1 = transform1
        self.transform2 = transform2
        self.prob = prob

    def __call__(self, img: Image.Image, target: dict):
        if random.random() < self.prob:
            return self.transform1(img, target)

        return self.transform2(img, target)
