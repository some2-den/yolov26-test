"""YOLO seg ラベルの読み込みヘルパー（evaluate.py / infer.py が使用）。"""
from __future__ import annotations

import os

import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")


def load_yolo_polygons(label_path, width: int, height: int):
    """YOLO seg ラベルを (クラスID, ポリゴン[px] Nx2 float) のリストで返す。"""
    polys = []
    if label_path is None or not os.path.isfile(label_path):
        return polys
    with open(label_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:  # class + 最低 3 点
                continue
            cls = int(float(parts[0]))
            coords = np.array(parts[1:], dtype=np.float64)
            if coords.size % 2 != 0:
                coords = coords[:-1]
            xy = coords.reshape(-1, 2)
            xy[:, 0] = np.clip(xy[:, 0] * width, 0, width)
            xy[:, 1] = np.clip(xy[:, 1] * height, 0, height)
            polys.append((cls, xy))
    return polys
