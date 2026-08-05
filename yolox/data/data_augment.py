#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import cv2
import numpy as np

import torch

from yolox.utils import xyxy2cxcywh

import math
import random


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Transform label coordinates
    n = len(targets)
    if n:
        # [DB-SORT-VIS] label 列 >= 10 时含可见框 [6:10], 需同步参与仿射变换
        has_vis = targets.shape[1] >= 10
        # warp points: 全身框 4 角 (x1y1, x2y2, x1y2, x2y1) + 可见框 4 角
        if has_vis:
            idx = [0, 1, 2, 3, 0, 3, 2, 1, 6, 7, 8, 9, 6, 9, 8, 7]
        else:
            idx = [0, 1, 2, 3, 0, 3, 2, 1]
        xy = np.ones((n * len(idx) // 2, 3))
        xy[:, :2] = targets[:, idx].reshape(xy.shape[0], 2)
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, len(idx))  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, len(idx))

        # create new boxes (全身框)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        full_xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        if has_vis:
            # create new boxes (可见框)
            xv = xy[:, [8, 10, 12, 14]]
            yv = xy[:, [9, 11, 13, 15]]
            vis_xy = np.concatenate((xv.min(1), yv.min(1), xv.max(1), yv.max(1))).reshape(4, n).T

        # clip boxes
        #xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        #xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates (基于全身框, 保持原有行为)
        i = box_candidates(box1=targets[:, :4].T * s, box2=full_xy.T)
        targets = targets[i]
        targets[:, :4] = full_xy[i]
        if has_vis:
            targets[:, 6:10] = vis_xy[i]

        targets = targets[targets[:, 0] < width]
        targets = targets[targets[:, 2] > 0]
        targets = targets[targets[:, 1] < height]
        targets = targets[targets[:, 3] > 0]

    return img, targets


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _mirror(image, boxes, vis_boxes=None):
    # [DB-SORT-VIS] 增加可选 vis_boxes 参数:
    #   传入时与 boxes 共享同一次翻转决策(保证可见框与全身框镜像一致);
    #   不传时行为与原始完全一致, 返回两个值。
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        if vis_boxes is not None:
            vis_boxes = vis_boxes.copy()
            vis_boxes[:, 0::2] = width - vis_boxes[:, 2::-2]
    if vis_boxes is None:
        return image, boxes
    return image, boxes, vis_boxes


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r, image            # [hgx0411] array for [C, H, W], resize_ratio, raw_image [H, W, C]


class TrainTransform:
    def __init__(self, p=0.5, rgb_means=None, std=None, max_labels=100):
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels = max_labels

    def __call__(self, image, targets, input_dim):
        # [DB-SORT-VIS] targets 列数 >= 10 时含可见框 [6:10](tlbr), 与全身框同步变换
        has_vis = targets.shape[1] >= 10
        boxes = targets[:, :4].copy()
        vis_boxes = targets[:, 6:10].copy() if has_vis else None
        labels = targets[:, 4].copy()
        ids = targets[:, 5].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 10 if has_vis else 6), dtype=np.float32)
            image, r_o, _ = preproc(image, input_dim, self.means, self.std)     # [hgx 0424] _ for raw_image
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        ids_o = targets_o[:, 5]
        vis_boxes_o = targets_o[:, 6:10] if has_vis else None
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)
        if has_vis:
            vis_boxes_o = xyxy2cxcywh(vis_boxes_o)

        image_t = _distort(image)
        if has_vis:
            image_t, boxes, vis_boxes = _mirror(image_t, boxes, vis_boxes)
        else:
            image_t, boxes = _mirror(image_t, boxes)
        height, width, _ = image_t.shape
        image_t, r_, _ = preproc(image_t, input_dim, self.means, self.std)      # [hgx 0424] _ for raw_image
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_
        if has_vis:
            vis_boxes = xyxy2cxcywh(vis_boxes)
            vis_boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        ids_t = ids[mask_b]
        vis_boxes_t = vis_boxes[mask_b] if has_vis else None

        if len(boxes_t) == 0:
            image_t, r_o, _ = preproc(image_o, input_dim, self.means, self.std)     # [hgx 0424] _ for raw_image
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            ids_t = ids_o
            if has_vis:
                vis_boxes_o *= r_o
                vis_boxes_t = vis_boxes_o

        labels_t = np.expand_dims(labels_t, 1)
        ids_t = np.expand_dims(ids_t, 1)

        if has_vis:
            # format: class_id, bbox(xywh), track_id, vis_bbox(xywh)
            targets_t = np.hstack((labels_t, boxes_t, ids_t, vis_boxes_t))
            padded_labels = np.zeros((self.max_labels, 10))
        else:
            # get new labels(targets), format: class_id, bbox(xywh), track_id
            targets_t = np.hstack((labels_t, boxes_t, ids_t))
            padded_labels = np.zeros((self.max_labels, 6))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1)):
        self.means = rgb_means
        self.swap = swap
        self.std = std

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _, raw_image = preproc(img, input_size, self.means, self.std, self.swap)
        return img, np.zeros((1, 5)), raw_image  # [hgx 0329] array of [C, H, W], zeros for targets, raw_image [H, W, C]
