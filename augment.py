"""YOLO seg 学習データのオフライン水増し（回転のみ）。

各画像を正方形にパディングしてから 90 / 180 / 270 度回転させ、画像とポリゴンに
同一の回転を適用する。90 度単位のロスレス変換なので、画像劣化・黒縁・
インスタンス欠落が一切生じない。元画像 1 枚 → 回転 3 枚 + 元 1 枚。

正方形パディングは、回転で縦横が入れ替わっても全画像が同一サイズになり、
バッチ化できるようにするため（正方形は 90 度回転で形が変わらない）。

※ 検証（正解）データには使わないこと。学習データのみに適用する。
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")


def read_label(path):
    """YOLO seg ラベルを (クラスID, Nx2 float[0-1]) のリストで返す。"""
    polys = []
    if not path or not os.path.isfile(path):
        return polys
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:  # class + 最低 3 点
                continue
            cls = int(float(parts[0]))
            coords = np.array(parts[1:], dtype=np.float64)
            if coords.size % 2 != 0:
                coords = coords[:-1]
            polys.append((cls, coords.reshape(-1, 2)))
    return polys


def write_label(path, polys):
    """ポリゴン群を YOLO seg 形式（正規化座標）で書き出す。"""
    lines = []
    for cls, xy in polys:
        coords = " ".join(f"{v:.6f}" for v in xy.reshape(-1))
        lines.append(f"{cls} {coords}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def pad_to_square(image, polys_px, size):
    """画像を size x size の中央へ配置し、ポリゴン座標も同じだけずらす。"""
    h, w = image.shape[:2]
    ox, oy = (size - w) // 2, (size - h) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[oy:oy + h, ox:ox + w] = image
    moved = []
    for cls, xy in polys_px:
        p = xy.copy()
        p[:, 0] += ox
        p[:, 1] += oy
        moved.append((cls, p))
    return canvas, moved


def rotate_points(xy, k, size):
    """size x size 画像内の点を 90*k 度回転（np.rot90 と整合）。"""
    x, y = xy[:, 0], xy[:, 1]
    if k == 1:    # 反時計回り 90 度
        return np.stack([y, size - 1 - x], axis=1)
    if k == 2:    # 180 度
        return np.stack([size - 1 - x, size - 1 - y], axis=1)
    if k == 3:    # 時計回り 90 度
        return np.stack([size - 1 - y, x], axis=1)
    return xy.copy()


def main():
    p = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    p.add_argument("--images", default=f"{home}/asset/train",
                   help="学習画像ディレクトリ（検証データには使わないこと）")
    p.add_argument("--labels", default=f"{home}/asset/train/data/labels/train")
    p.add_argument("--output", default=f"{home}/asset/train_aug")
    p.add_argument("--no-original", action="store_true",
                   help="元画像（回転なし）を出力に含めない")
    args = p.parse_args()

    out_img = os.path.join(args.output, "images")
    out_lbl = os.path.join(args.output, "labels")
    # 過去の出力が混ざらないよう作り直す
    for d in (out_img, out_lbl):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(args.images, f"*{ext}")))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"画像が見つかりません: {args.images}")

    # 全画像を収める正方形サイズ（32 の倍数に切り上げ）
    max_dim = 0
    for path in paths:
        with Image.open(path) as im:
            max_dim = max(max_dim, im.width, im.height)
    size = ((max_dim + 31) // 32) * 32
    print(f"正方形パディングサイズ: {size}x{size}")

    # k=0 は回転なし（元画像をパディングしたもの）
    rotations = [(0, "")] if args.no_original else [(0, "")]
    rotations += [(1, "_rot090"), (2, "_rot180"), (3, "_rot270")]

    n_src, n_out = 0, 0
    cls_count = {}
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        ext = os.path.splitext(path)[1]
        image = np.array(Image.open(path).convert("RGB"))
        h, w = image.shape[:2]

        polys = read_label(os.path.join(args.labels, f"{stem}.txt"))
        polys_px = [(c, xy * [w, h]) for c, xy in polys]
        padded, polys_px = pad_to_square(image, polys_px, size)
        n_src += 1

        for k, suffix in rotations:
            if k == 0 and args.no_original:
                continue
            rot_img = np.rot90(padded, k) if k else padded
            rot_polys = []
            for cls, xy in polys_px:
                rp = rotate_points(xy, k, size) / size
                rot_polys.append((cls, rp))
                cls_count[cls] = cls_count.get(cls, 0) + 1

            name = f"{stem}{suffix}"
            Image.fromarray(np.ascontiguousarray(rot_img)).save(
                os.path.join(out_img, f"{name}{ext}"))
            write_label(os.path.join(out_lbl, f"{name}.txt"), rot_polys)
            n_out += 1

    print(f"元画像 {n_src} 枚 -> 出力 {n_out} 枚"
          f"（元 + 90/180/270 度回転）")
    print(f"クラス別インスタンス数: "
          + ", ".join(f"{k}:{v}" for k, v in sorted(cls_count.items())))
    print(f"出力先: {out_img}\n        {out_lbl}")


if __name__ == "__main__":
    main()
