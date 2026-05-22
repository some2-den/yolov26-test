"""学習済み YOLO26-seg で推論し、結果を可視化する。

正解ラベルが存在する画像については、正解と予測を並べた比較画像を保存し、
さらに全画像分の混同行列・IoU・クラス別指標をまとめて出力する。
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset import IMG_EXTS
from evaluate import (compute_metrics, gt_masks, match, predict_instances,
                      report, save_outputs)

# クラスごとの表示色（No / Minor / Major / Blue sheet）
COLORS = [(0, 200, 0), (255, 200, 0), (220, 0, 0), (0, 120, 255)]


def load_font(size=18):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_instances(image, instances, id2label, font):
    """インスタンス群（{mask, label_id, score}）を画像に重ね描きする。"""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for inst in instances:
        mask = inst["mask"]
        if not mask.any():
            continue
        color = COLORS[inst["label_id"] % len(COLORS)]
        colored = np.zeros((*mask.shape, 4), dtype=np.uint8)
        colored[mask] = (*color, 110)
        overlay = Image.alpha_composite(overlay, Image.fromarray(colored, "RGBA"))
    draw = ImageDraw.Draw(overlay)
    for inst in instances:
        mask = inst["mask"]
        if not mask.any():
            continue
        color = COLORS[inst["label_id"] % len(COLORS)]
        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        name = id2label[inst["label_id"]]
        text = name if inst.get("score") is None else f"{name} {inst['score']:.2f}"
        tw = draw.textlength(text, font=font)
        draw.rectangle([cx, cy - 20, cx + tw + 6, cy], fill=(*color, 230))
        draw.text((cx + 3, cy - 19), text, fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(base, overlay).convert("RGB")


def side_by_side(left, right, left_title, right_title, font):
    """2 枚の画像を見出し付きで横並びにする。"""
    w, h = left.size
    gap, top = 12, 32
    canvas = Image.new("RGB", (w * 2 + gap, h + top), (255, 255, 255))
    canvas.paste(left, (0, top))
    canvas.paste(right, (w + gap, top))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), left_title, fill=(0, 0, 0), font=font)
    draw.text((w + gap + 8, 7), right_title, fill=(0, 0, 0), font=font)
    return canvas


def main():
    home = os.path.expanduser("~")
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=f"{home}/yolov26/outputs/train/weights/best.pt")
    p.add_argument("--images", default=f"{home}/asset/validate")
    p.add_argument("--labels", default=f"{home}/asset/validate/data/labels/train",
                   help="正解ラベルのディレクトリ（比較画像・混同行列に使用）")
    p.add_argument("--output", default=f"{home}/yolov26/outputs/predictions")
    p.add_argument("--threshold", type=float, default=0.5, help="スコア閾値")
    p.add_argument("--iou-thr", type=float, default=0.5, help="マッチングの IoU 閾値")
    args = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    id2label = {int(k): v for k, v in model.names.items()}
    n_cls = len(id2label)
    labels = [id2label[i] for i in range(n_cls)]
    font = load_font(18)
    title_font = load_font(20)
    os.makedirs(args.output, exist_ok=True)

    paths = []
    if os.path.isdir(args.images):
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(args.images, f"*{ext}")))
    else:
        paths = [args.images]
    paths = sorted(set(paths))
    print(f"{len(paths)} 枚を推論")

    bg = n_cls
    cm = np.zeros((n_cls + 1, n_cls + 1), dtype=np.int64)
    iou_by_class = {c: [] for c in range(n_cls)}
    all_ious = []

    for path in paths:
        image = Image.open(path).convert("RGB")
        w, h = image.size
        stem = os.path.splitext(os.path.basename(path))[0]

        preds = predict_instances(model, path, args.threshold, (w, h))
        pred_inst = [{"mask": m, "label_id": c, "score": s} for c, s, m in preds]
        pred_vis = draw_instances(image, pred_inst, id2label, font)

        # 正解ラベルがあれば比較画像と混同行列を作る
        label_path = os.path.join(args.labels, f"{stem}.txt")
        if os.path.isfile(label_path):
            gts = gt_masks(label_path, w, h)
            gt_inst = [{"mask": m, "label_id": c, "score": None} for c, m in gts]
            gt_vis = draw_instances(image, gt_inst, id2label, font)

            matched, fp, fn = match(preds, gts, args.iou_thr)
            img_ious = []
            for pi, gi, iou in matched:
                gt_cls = gts[gi][0]
                cm[gt_cls, preds[pi][0]] += 1
                iou_by_class[gt_cls].append(iou)
                all_ious.append(iou)
                img_ious.append(iou)
            for pi in fp:
                cm[bg, preds[pi][0]] += 1
            for gi in fn:
                cm[gts[gi][0], bg] += 1

            mean_iou = float(np.mean(img_ious)) if img_ious else 0.0
            compare = side_by_side(
                gt_vis, pred_vis, f"Ground Truth ({len(gts)})",
                f"Prediction ({len(pred_inst)})  IoU={mean_iou:.2f}", title_font)
            out_path = os.path.join(args.output, f"{stem}_compare.jpg")
            compare.save(out_path)
            print(f"  {stem}: GT {len(gts)} / 予測 {len(pred_inst)}  "
                  f"IoU={mean_iou:.2f} -> {os.path.basename(out_path)}")
        else:
            out_path = os.path.join(args.output, f"{stem}_pred.jpg")
            pred_vis.save(out_path)
            print(f"  {stem}: 予測 {len(pred_inst)}（正解なし）-> "
                  f"{os.path.basename(out_path)}")

    # 全画像分の混同行列・IoU・クラス別指標をまとめて出力
    if all_ious or cm.sum() > 0:
        metrics = compute_metrics(cm, iou_by_class, n_cls)
        print(f"\n=== 推論時評価  IoU閾値={args.iou_thr}  "
              f"スコア閾値={args.threshold} ===")
        report(cm, metrics, labels, all_ious)
        save_outputs(cm, labels, args.output)
    print(f"\n出力先: {args.output}")


if __name__ == "__main__":
    main()
