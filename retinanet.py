import json
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.ops import batched_nms, sigmoid_focal_loss

import modules.transforms as T
from modules.datasets import CocoDetection
from modules.models import RetinaNet
from modules.utils import calc_iou, collate_fn, convert_to_xywh, convert_to_xyxy, generate_subset
from tqdm import tqdm
from collections import deque

@dataclass
class Config:
    img_dir: str = "./data/coco/val2014"
    annot_file: str = "./data/coco/instances_val2014_small.json"

    train_ratio: float = 0.8
    num_epochs: int = 10
    lr_drop: int = 45
    val_interval: int = 1
    lr: float = 1e-5
    gradient_clip: float = 0.1
    batch_size: int = 8
    num_workers: int = 4


@torch.no_grad()
def post_process(
    preds_class: torch.Tensor,
    preds_box: torch.Tensor,
    anchors: torch.Tensor,
    targets: list[dict],
    conf_threshold: float = 0.05,
    nms_threshold: float = 0.5,
):
    anchors_xywh = convert_to_xywh(anchors)

    preds_box[:, :, :2] = anchors_xywh[:, :2] + preds_box[:, :, :2] * anchors_xywh[:, 2:]
    preds_box[:, :, 2:] = preds_box[:, :, 2:].exp() * anchors_xywh[:, 2:]

    preds_box = convert_to_xyxy(preds_box)

    preds_class = preds_class.sigmoid()

    scores = []
    labels = []
    boxes = []
    for img_preds_class, img_preds_box, img_targets in zip(preds_class, preds_box, targets):
        # Clamp bounding box into image size.
        img_preds_box[:, ::2] = img_preds_box[:, ::2].clamp(
            min=0, max=img_targets["img_size"][0]
        )  # x
        img_preds_box[:, 1::2] = img_preds_box[:, 1::2].clamp(
            min=0, max=img_targets["img_size"][1]
        )  # y

        # Rescale bounding box to fit with original image.
        img_preds_box *= img_targets["orig_img_size"][0] / img_targets["img_size"][0]

        img_preds_score, img_preds_label = img_preds_class.max(dim=1)

        keep = img_preds_score > conf_threshold
        img_preds_score = img_preds_score[keep]
        img_preds_label = img_preds_label[keep]
        img_preds_box = img_preds_box[keep]

        # Apply NMS per class
        keep_indices = batched_nms(img_preds_box, img_preds_score, img_preds_label, nms_threshold)
        scores.append(img_preds_score[keep_indices])
        labels.append(img_preds_label[keep_indices])
        boxes.append(img_preds_box[keep_indices])

    return scores, labels, boxes


def loss_fn(
    preds_class: torch.Tensor,
    preds_box: torch.Tensor,
    anchors: torch.Tensor,
    targets: list[dict],
    iou_lower_threshold: float = 0.4,
    iou_upper_threshold: float = 0.5,
):
    """Compute Focal Loss.
    Args:
        preds_class (Tensor[N, num_anchors, num_classes]): Classes.
        preds_box (Tensor[N, num_anchors, 4]): Bounding boxes.
            Coordinate should be (x_diff, y_diff, w_diff, h_diff).
        anchors (Tensor[num_anchors, 4]): Coordinate should be (xmin, ymin, xmax, ymax).
        targets: Labels.
    """
    anchors_xywh = convert_to_xywh(anchors)

    # Calculate target function per image
    loss_class = preds_class.new_tensor(0)
    loss_box = preds_class.new_tensor(0)
    for img_preds_class, img_preds_box, img_targets in zip(preds_class, preds_box, targets):
        # If no ground truth for this image.
        if img_targets["classes"].shape[0] == 0:
            # Create target class as background.
            targets_class = torch.zeros_like(img_preds_class)
            loss_class += sigmoid_focal_loss(img_preds_class, targets_class, reduction="sum")
            continue

        # Get a bounding box which has max IoU.
        ious = calc_iou(anchors, img_targets["boxes"])[0]
        ious_max, ious_argmax = ious.max(dim=1)

        # Init class label as -1.
        # Set label of anchor box as -1 if iou_lower_threshold <= IoU <= iou_upper_threshold
        # in order not to calculate loss.
        targets_class = torch.full_like(img_preds_class, -1)

        targets_class[ious_max < iou_lower_threshold] = 0

        # If IoU > iou_upper_threshold, set as classification/regression target.
        positive_masks = ious_max > iou_upper_threshold
        num_positive_anchors = positive_masks.sum()

        targets_class[positive_masks] = 0
        assigned_classes = img_targets["classes"][ious_argmax]
        targets_class[positive_masks, assigned_classes[positive_masks]] = 1

        loss_class += (
            (targets_class != -1) * sigmoid_focal_loss(img_preds_class, targets_class)
        ).sum() / num_positive_anchors.clamp(min=1)

        # If no positive anchors, skip calculation of loss_box
        if num_positive_anchors == 0:
            continue

        # Get ground truth per anchor
        assgined_boxes = img_targets["boxes"][ious_argmax]
        assgined_boxes_xywh = convert_to_xywh(assgined_boxes)

        targets_box = torch.zeros_like(img_preds_box)
        targets_box[:, :2] = (assgined_boxes_xywh[:, :2] - anchors_xywh[:, :2]) / anchors_xywh[
            :, 2:
        ]
        targets_box[:, 2:] = (assgined_boxes_xywh[:, 2:] / anchors_xywh[:, 2:]).log()
        loss_box += F.smooth_l1_loss(
            img_preds_box[positive_masks], targets_box[positive_masks], beta=1 / 9
        )

    batch_size = preds_class.shape[0]
    loss_class /= batch_size
    loss_box /= batch_size

    return loss_class, loss_box


def get_loader(root_dir, annot_file, batch_size=16, train_ratio: float = 0.8, num_workers: int = 4):
    min_sizes = (480, 512, 544, 576, 608)
    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(min_sizes, max_size=1024),
                T.Compose(
                    [
                        T.RandomSizeCrop(scale=(0.8, 1.0), ratio=(0.75, 1.333)),
                        T.RandomResize(min_sizes, max_size=1024),
                    ]
                ),
            ),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_transform = T.Compose(
        [
            T.RandomResize([min_sizes[-1]], max_size=1024),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = CocoDetection(root_dir, annot_file, train_transform)
    val_dataset = CocoDetection(root_dir, annot_file, val_transform)
    train_set, val_set = generate_subset(train_dataset, train_ratio)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=SubsetRandomSampler(train_set),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=val_set,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


class CocoEvaluator:
    metric_names = (
        "mAP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "mAR",
        "AR_small",
        "AR_medium",
        "AR_large",
    )

    def __init__(self, dataset: CocoDetection):
        self.dataset = dataset
        self.reset()

    def reset(self):
        self.preds = []

    def update(self, preds):
        """
        Args:
            preds: image_id, category_id, score, bbox
        """
        for p in preds:
            p["category_id"] = self.dataset.preds_to_coco[p["category_id"]]
        self.preds.extend(preds)

    def get_metrics(self):
        if len(self.preds) == 0:
            return {}
        with open("tmp.json", "w") as f:
            json.dump(self.preds, f)

        coco_results = self.dataset.coco.loadRes("tmp.json")
        coco_eval = COCOeval(self.dataset.coco, coco_results, "bbox")
        coco_eval.params.imgIds = list(map(lambda x: x["image_id"], self.preds))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {metric: coco_eval.stats[i] for i, metric in enumerate(self.metric_names)}
        return metrics


class RetinaNetModule(pl.LightningModule):
    def __init__(
        self, num_classes: int, learning_rate: float, lr_drop: float, evaluator: CocoEvaluator
    ):
        super().__init__()
        self.model = RetinaNet(num_classes)
        self.learning_rate = learning_rate
        self.lr_drop = lr_drop
        self.evaluator = evaluator

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[self.lr_drop], gamma=0.1
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch):
        imgs, targets = batch
        preds_class, preds_box, anchors = self.model(imgs)
        loss_class, loss_box = loss_fn(preds_class, preds_box, anchors, targets)
        loss = loss_class + loss_box
        self.log("train_loss_class", loss_class)
        self.log("train_loss_box", loss_box)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        imgs, targets = batch
        preds_class, preds_box, anchors = self.model(imgs)
        loss_class, loss_box = loss_fn(preds_class, preds_box, anchors, targets)
        loss = loss_class + loss_box
        self.log("val_loss_class", loss_class)
        self.log("val_loss_box", loss_box)
        self.log("val_loss", loss, prog_bar=True)

        scores, labels, boxes = post_process(preds_class, preds_box, anchors, targets)
        preds = []
        for img_scores, img_labels, img_boxes, img_targets in zip(scores, labels, boxes, targets):
            img_boxes[:, 2:] -= img_boxes[:, :2]  # to xywh
            for score, label, box in zip(img_scores, img_labels, img_boxes):
                preds.append(
                    {
                        "image_id": img_targets["image_id"].item(),
                        "category_id": label.item(),
                        "score": score.item(),
                        "bbox": box.cpu().numpy().tolist(),
                    }
                )
        self.evaluator.update(preds)
        return loss

    def on_validation_epoch_end(self) -> None:
        metrics = self.evaluator.get_metrics()
        self.evaluator.reset()
        self.log_dict({f"val_{key}": value for key, value in metrics.items()})


def main():
    config = Config()
    train_loader, val_loader = get_loader(
        config.img_dir, config.annot_file, config.batch_size, config.train_ratio, config.num_workers
    )
    model_module = RetinaNetModule(
        num_classes=2,
        learning_rate=config.lr,
        lr_drop=config.lr_drop,
        evaluator=CocoEvaluator(val_loader.dataset),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=config.num_epochs,
        devices=1,
        gradient_clip_val=config.gradient_clip,
    )
    trainer.fit(model_module, train_dataloaders=train_loader, val_dataloaders=val_loader)





def train_eval():
    config = Config()

    # データ拡張・整形クラスの設定
    min_sizes = (480, 512, 544, 576, 608)
    train_transforms = T.Compose((
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(min_sizes, max_size=1024),
            T.Compose((
                T.RandomSizeCrop(scale=(0.8, 1.0),
                                 ratio=(0.75, 1.333)),
                T.RandomResize(min_sizes, max_size=1024),
            ))
        ),
        T.ToTensor(),
        # ImageNetデータセットの平均と標準偏差
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ))
    test_transforms = T.Compose((
        # テストは短辺最大で実行
        T.RandomResize((min_sizes[-1],), max_size=1024),
        T.ToTensor(),
        # ImageNetデータセットの平均と標準偏差
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ))

    # データセットの用意
    train_dataset = CocoDetection(
        config.img_dir,
        config.annot_file, transform=train_transforms)
    val_dataset = CocoDetection(
        config.img_dir,
        config.annot_file, transform=test_transforms)

    # Subset samplerの生成
    train_set, val_set = generate_subset(
        train_dataset, config.train_ratio)

    print(f'学習セットのサンプル数: {len(train_set)}')
    print(f'検証セットのサンプル数: {len(val_set)}')

    # 学習時にランダムにサンプルするためのサンプラー
    train_sampler = SubsetRandomSampler(train_set)

    # DataLoaderを生成
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, sampler=train_sampler,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, sampler=val_set,
        collate_fn=collate_fn)

    # RetinaNetの生成
    model = RetinaNet(len(train_dataset.classes))
    # ResNet18をImageNetの学習済みモデルで初期化
    # 最後の全結合層がないなどのモデルの改変を許容するため、strict=False
    model.backbone.load_state_dict(torch.hub.load_state_dict_from_url(
        'https://download.pytorch.org/models/resnet18-5c106cde.pth'),
                                   strict=False)

    # モデルを指定デバイスに転送
    model.to("cuda")

    # Optimizerの生成
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # 指定したエポックで学習率を1/10に減衰するスケジューラを生成
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[config.lr_drop], gamma=0.1)

    for epoch in range(config.num_epochs):
        model.train()

        with tqdm(train_loader) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')

            # 移動平均計算用
            losses_class = deque()
            losses_box = deque()
            losses = deque()
            for imgs, targets in pbar:
                imgs = imgs.to("cuda")
                targets = [{k: v.to("cuda")
                            for k, v in target.items()}
                           for target in targets]

                optimizer.zero_grad()

                preds_class, preds_box, anchors = model(imgs)

                loss_class, loss_box = loss_fn(
                    preds_class, preds_box, anchors, targets)
                loss = loss_class + loss_box

                loss.backward()

                # 勾配全体のL2ノルムが上限を超えるとき上限値でクリップ
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip)

                optimizer.step()

                losses_class.append(loss_class.item())
                losses_box.append(loss_box.item())
                losses.append(loss.item())
                if len(losses) > 100:
                    losses_class.popleft()
                    losses_box.popleft()
                    losses.popleft()
                pbar.set_postfix({
                    'loss': torch.Tensor(losses).mean().item(),
                    'loss_class': torch.Tensor(
                        losses_class).mean().item(),
                    'loss_box': torch.Tensor(
                        losses_box).mean().item()})

        # スケジューラでエポック数をカウント
        scheduler.step()

        # パラメータを保存
#        torch.save(model.state_dict(), config.save_file)

        # 検証
        if (epoch + 1) % config.val_interval == 0:
            evaluate(val_loader, model, loss_fn)

def evaluate(data_loader: DataLoader, model: torch.nn.Module,
             loss_func, conf_threshold: float=0.05,
             nms_threshold: float=0.5):
    model.eval()

    losses_class = []
    losses_box = []
    losses = []
    preds = []
    img_ids = []
    for imgs, targets in tqdm(data_loader, desc='[Validation]'):
        with  torch.no_grad():
            imgs = imgs.to("cuda")
            targets = [{k: v.to("cuda")
                        for k, v in target.items()}
                       for target in targets]

            preds_class, preds_box, anchors = model(imgs)

            loss_class, loss_box = loss_func(
                preds_class, preds_box, anchors, targets)
            loss = loss_class + loss_box

            losses_class.append(loss_class)
            losses_box.append(loss_box)
            losses.append(loss)

            # 後処理により最終的な検出矩形を取得
            scores, labels, boxes = post_process(
                preds_class, preds_box, anchors, targets,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold)

            for img_scores, img_labels, img_boxes, img_targets in zip(
                    scores, labels, boxes, targets):
                img_ids.append(img_targets['image_id'].item())

                # 評価のためにCOCOの元々の矩形表現である
                # xmin, ymin, width, heightに変換
                img_boxes[:, 2:] -= img_boxes[:, :2]

                for score, label, box in zip(
                        img_scores, img_labels, img_boxes):
                    # COCO評価用のデータの保存
                    preds.append({
                        'image_id': img_targets['image_id'].item(),
                        'category_id': \
                        data_loader.dataset.to_coco_label(
                            label.item()),
                        'score': score.item(),
                        'bbox': box.to('cpu').numpy().tolist()
                    })

    loss_class = torch.stack(losses_class).mean().item()
    loss_box = torch.stack(losses_box).mean().item()
    loss = torch.stack(losses).mean().item()
    print(f'Validation loss = {loss:.3f},'
          f'class loss = {loss_class:.3f}, '
          f'box loss = {loss_box:.3f} ')

    if len(preds) == 0:
        print('Nothing detected, skip evaluation.')

        return

    # COCOevalクラスを使って評価するには検出結果を
    # jsonファイルに出力する必要があるため、jsonファイルに一時保存
    with open('tmp.json', 'w') as f:
        json.dump(preds, f)

    # 一時保存した検出結果をCOCOクラスを使って読み込み
    coco_results = data_loader.dataset.coco.loadRes('tmp.json')

    # COCOevalクラスを使って評価
    coco_eval = COCOeval(
        data_loader.dataset.coco, coco_results, 'bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    # main()
    train_eval()
