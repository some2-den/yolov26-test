"""学習済み YOLO26-seg を検証データで評価し、混同行列と各種指標を出力する。

予測インスタンスと正解インスタンスを IoU で対応付け（クラス依存の貪欲マッチング）、
(クラス数+1)x(クラス数+1) の混同行列を作る。
末尾の行/列は「背景」= 未検出(FN) / 誤検出(FP) を表す。
クラス別に TP / FP / FN と precision / recall / IoU をまとめて出力する。
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from dataset import IMG_EXTS, load_yolo_polygons


def gt_masks(label_path, w, h):
    """正解ポリゴンを (クラスID, bool マスク) のリストへ。"""
    import cv2
    out = []
    for cls, xy in load_yolo_polygons(label_path, w, h):
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [xy.round().astype(np.int32)], color=1)
        out.append((cls, m.astype(bool)))
    return out


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


def match(preds, gts, iou_thr):
    """予測と正解を IoU 貪欲マッチング。(マッチ対, 未マッチpred, 未マッチgt) を返す。

    preds: [(cls, score, mask)] / gts: [(cls, mask)]
    マッチ対は (pred_idx, gt_idx, iou)。

    戦略:
    1) 予測を score 降順で走査
    2) 同一クラスの未使用 GT のみ候補
    3) IoU が最大かつ iou_thr 以上の 1 件に割り当て
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    used_gt = set()
    matched, fp = [], []
    for pi in order:
        pred_cls = preds[pi][0]
        best_iou, best_gt = iou_thr, -1
        for gi, (gt_cls, gmask) in enumerate(gts):
            if gi in used_gt:
                continue
            if gt_cls != pred_cls:
                continue
            iou = mask_iou(preds[pi][2], gmask)
            if iou >= best_iou:
                best_iou, best_gt = iou, gi
        if best_gt >= 0:
            used_gt.add(best_gt)
            matched.append((pi, best_gt, best_iou))
        else:
            fp.append(pi)
    fn = [gi for gi in range(len(gts)) if gi not in used_gt]
    return matched, fp, fn


def compute_metrics(cm, iou_by_class, n_cls):
    """混同行列とクラス別 IoU リストから指標 dict を作る。"""
    metrics = {}
    for c in range(n_cls):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        ious = iou_by_class.get(c, [])
        metrics[c] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "iou": float(np.mean(ious)) if ious else 0.0,
            "matches": len(ious),
        }
    return metrics


def plot_confusion(cm, labels, out_path, normalize=True):
    """混同行列をヒートマップ画像として保存。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = cm.astype(float)
    title = "Confusion Matrix"
    if normalize:
        col = mat.sum(axis=0, keepdims=True)
        mat = np.divide(mat, col, out=np.zeros_like(mat), where=col > 0)
        title += " (column-normalized)"

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() or 1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{mat[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def report(cm, metrics, labels, all_ious):
    """混同行列とクラス別指標（TP/FP/FN + precision/recall/IoU）を表示。"""
    names = labels + ["background"]
    w = max(len(n) for n in names) + 2
    print("\n混同行列（行=正解 / 列=予測, 単位=インスタンス数）")
    print(" " * w + "".join(f"{n:>14}" for n in names))
    for i, row in enumerate(cm):
        print(f"{names[i]:<{w}}" + "".join(f"{int(v):>14}" for v in row))

    print("\nクラス別指標（生カウント + 率）")
    print(f"{'class':<{w}}{'TP':>7}{'FP':>7}{'FN':>7}"
          f"{'precision':>11}{'recall':>10}{'IoU':>9}{'match':>8}")
    for c, m in metrics.items():
        print(f"{labels[c]:<{w}}{m['tp']:>7}{m['fp']:>7}{m['fn']:>7}"
              f"{m['precision']:>11.3f}{m['recall']:>10.3f}{m['iou']:>9.3f}"
              f"{m['matches']:>8}")
    mp = np.mean([m["precision"] for m in metrics.values()])
    mr = np.mean([m["recall"] for m in metrics.values()])
    mi = np.mean([m["iou"] for m in metrics.values()])
    sum_tp = int(sum(m["tp"] for m in metrics.values()))
    sum_fp = int(sum(m["fp"] for m in metrics.values()))
    sum_fn = int(sum(m["fn"] for m in metrics.values()))
    print(f"{'total/mean':<{w}}{sum_tp:>7}{sum_fp:>7}{sum_fn:>7}"
          f"{mp:>11.3f}{mr:>10.3f}{mi:>9.3f}{len(all_ious):>8}")
    overall_iou = float(np.mean(all_ious)) if all_ious else 0.0
    print(f"\n全マッチインスタンスの平均 mask IoU: {overall_iou:.3f}  "
          f"(マッチ数 {len(all_ious)})")


def save_outputs(cm, labels, output_dir):
    """混同行列を PNG と CSV で保存。"""
    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion(cm, labels + ["background"], png, normalize=True)
    csv = os.path.join(output_dir, "confusion_matrix.csv")
    rows = [",".join([""] + labels + ["background"])]
    for i, name in enumerate(labels + ["background"]):
        rows.append(",".join([name] + [str(int(v)) for v in cm[i]]))
    with open(csv, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"混同行列を保存: {png}\n            : {csv}")
    return png, csv


def predict_instances(model, image_path, score_thr, size):
    """YOLO26 で 1 枚を推論し [(クラスID, スコア, bool マスク)] を返す。"""
    import cv2
    w, h = size
    res = model.predict(source=image_path, conf=score_thr,
                        retina_masks=True, verbose=False)[0]
    preds = []
    if res.masks is None:
        return preds
    masks = res.masks.data.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    for i in range(len(cls)):
        m = masks[i]
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_NEAREST)
        m = m > 0.5
        if m.any():
            preds.append((int(cls[i]), float(conf[i]), m))
    return preds


def evaluate_images(model, paths, label_dir, n_cls, iou_thr, score_thr):
    """画像群を評価し (混同行列, クラス別IoUリスト, 全IoUリスト) を返す。"""
    from PIL import Image

    bg = n_cls
    cm = np.zeros((n_cls + 1, n_cls + 1), dtype=np.int64)
    iou_by_class = {c: [] for c in range(n_cls)}
    all_ious = []
    for path in paths:
        with Image.open(path) as im:
            w, h = im.size
        stem = os.path.splitext(os.path.basename(path))[0]
        gts = gt_masks(os.path.join(label_dir, f"{stem}.txt"), w, h)
        preds = predict_instances(model, path, score_thr, (w, h))

        matched, fp, fn = match(preds, gts, iou_thr)
        for pi, gi, iou in matched:
            gt_cls = gts[gi][0]
            cm[gt_cls, preds[pi][0]] += 1
            iou_by_class[gt_cls].append(iou)
            all_ious.append(iou)
        for pi in fp:
            cm[bg, preds[pi][0]] += 1   # 正解=背景, 予測=クラス -> 誤検出
        for gi in fn:
            cm[gts[gi][0], bg] += 1     # 正解=クラス, 予測=背景 -> 未検出
    return cm, iou_by_class, all_ious


def run(weights, image_dir, label_dir, output_dir,
        iou_thr=0.5, score_thr=0.5):
    """評価本体。混同行列(np.ndarray)と指標 dict を返す。"""
    from ultralytics import YOLO

    model = YOLO(weights)
    id2label = {int(k): v for k, v in model.names.items()}
    n_cls = len(id2label)
    labels = [id2label[i] for i in range(n_cls)]

    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
    paths = sorted(set(paths))

    cm, iou_by_class, all_ious = evaluate_images(
        model, paths, label_dir, n_cls, iou_thr, score_thr)
    metrics = compute_metrics(cm, iou_by_class, n_cls)

    print(f"\n=== 評価: {weights} ===")
    print(f"画像 {len(paths)} 枚  IoU閾値={iou_thr}  スコア閾値={score_thr}")
    report(cm, metrics, labels, all_ious)
    save_outputs(cm, labels, output_dir)
    return cm, metrics


def main():
    home = os.path.expanduser("~")
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=f"{home}/yolov26/outputs/train/weights/best.pt")
    p.add_argument("--images", default=f"{home}/asset/validate")
    p.add_argument("--labels", default=f"{home}/asset/validate/data/labels/train")
    p.add_argument("--output-dir", default=f"{home}/yolov26/outputs")
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--score-thrs", type=float, nargs="+", default=None,
                   help="複数スコア閾値で評価する場合に指定（例: --score-thrs 0.25 0.5）")
    args = p.parse_args()
    if args.score_thrs:
        for score_thr in args.score_thrs:
            thr_text = f"{score_thr:.6f}".rstrip("0").rstrip(".")
            thr_tag = thr_text.replace(".", "_")
            out_dir = os.path.join(args.output_dir, f"score_thr_{thr_tag}")
            run(args.weights, args.images, args.labels, out_dir,
                args.iou_thr, score_thr)
    else:
        run(args.weights, args.images, args.labels, args.output_dir,
            args.iou_thr, args.score_thr)


if __name__ == "__main__":
    main()
