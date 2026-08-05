# encoding: utf-8
"""
[DB-SORT-VIS] CrowdHuman 双回归头(全身框 + 可见框)训练配置。

数据准备(见 tools/convert_crowdhuman_to_coco.py):
  <get_yolox_datadir()>/crowdhuman/
  ├── CrowdHuman_train/            # 官方图片目录(原样)
  ├── CrowdHuman_val/
  ├── annotation_train.odgt        # 官方标注
  ├── annotation_val.odgt
  └── annotations/                 # 转换脚本自动生成
      ├── train.json
      └── val.json

训练命令:
  python tools/train.py -f exps/example/mot/yolox_crowdhuman_dual.py -d 1 -b 16 --fp16 -o
"""
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ---------- 数据集配置 (CrowdHuman) ----------
        self.data_dir_name = "crowdhuman"   # 位于 get_yolox_datadir() 下
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        # ---------- 训练超参数 ----------
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)
        self.max_epoch = 80
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

        # ---------- [DB-SORT-VIS] 可见框双头 ----------
        self.use_visible_head = True
        self.vis_loss_weight = 0.7        # 可见框回归损失权重 (loss 项)
        self.vis_assign_weight = 0.3      # SimOTA cost 中可见框 IoU 项权重
        self.vis_warmup_steps = 5000      # 分配权重热身步数

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), self.data_dir_name),
            json_file=self.train_ann,
            name='CrowdHuman_train',      # CrowdHuman 官方图片子目录名
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,          # CrowdHuman 密集场景, 放宽标签上限
            ),
            use_visible_head=self.use_visible_head,  # [DB-SORT-VIS]
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, run_tracking=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), self.data_dir_name),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='CrowdHuman_val',
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            run_tracking=run_tracking
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev, run_tracking=False)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
