"""YOLO 形式のデータセット構造を symlink で用意し、data.yaml を書き出す。

学習データは回転水増し済みの ~/asset/train_aug（augment.py の出力）、
検証データは ~/asset/validate を使う。Ultralytics は画像パスの "images" を
"labels" に置換してラベルを探すため、その規約に沿った構造を作る。
"""
from __future__ import annotations

import glob
import os
import shutil

HOME = os.path.expanduser("~")
ROOT = f"{HOME}/yolov26/dataset"
DATA_YAML = f"{HOME}/yolov26/data.yaml"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")
NAMES = {0: "No damage", 1: "Minor damage", 2: "Major damage", 3: "Blue sheet"}

# (種別, 分割) -> 元データの場所
SOURCES = {
    ("images", "train"): f"{HOME}/asset/train_aug/images",
    ("labels", "train"): f"{HOME}/asset/train_aug/labels",
    ("images", "val"): f"{HOME}/asset/validate",
    ("labels", "val"): f"{HOME}/asset/validate/data/labels/train",
}


def link_files(src, dst, exts):
    """src 内の該当拡張子ファイルを dst へ symlink する（dst は作り直す）。"""
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(src, f"*{e}")))
    for f in sorted(set(files)):
        os.symlink(os.path.abspath(f), os.path.join(dst, os.path.basename(f)))
    return len(files)


def main():
    for (kind, split), src in SOURCES.items():
        if not os.path.isdir(src):
            hint = "（augment.py を先に実行してください）" if "train_aug" in src else ""
            raise FileNotFoundError(f"元データがありません: {src} {hint}")
        exts = IMG_EXTS if kind == "images" else (".txt",)
        n = link_files(src, os.path.join(ROOT, kind, split), exts)
        print(f"  {kind}/{split}: {n} ファイル <- {src}")

    with open(DATA_YAML, "w") as f:
        f.write(f"path: {ROOT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for k, v in NAMES.items():
            f.write(f"  {k}: {v}\n")
    print(f"  data.yaml: {DATA_YAML}")
    return DATA_YAML


if __name__ == "__main__":
    main()
