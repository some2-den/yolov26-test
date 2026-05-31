"""学習済み YOLO26-seg を検証データで評価し、混同行列と各種指標を出力する。

予測インスタンスと正解インスタンスを IoU で対応付ける（クラス依存の貪欲法）。
マッチ対象は同一クラスの未使用 GT のみに限定し、予測スコア降順で 1:1 対応を作る。
(クラス数+1)x(クラス数+1) の混同行列を作り、末尾の行/列は「背景」=
未検出(FN) / 誤検出(FP) を表す。
クラス別に precision / recall / IoU と TP/FP/FN をまとめて出力する。
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


def ap_from_pr(recalls, precisions):
    """precision-recall 曲線から AP を求める。"""
    if len(recalls) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], np.asarray(recalls), [1.0]))
    mpre = np.concatenate(([0.0], np.asarray(precisions), [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def init_map_state(n_cls):
    """mAP 計算用の予測/GT 集計器を初期化する。"""
    return {
        "preds": {c: [] for c in range(n_cls)},
        "gts": {c: {} for c in range(n_cls)},
    }


def update_map_state(map_state, image_key, preds, gts):
    """1 枚分の予測と GT を mAP 集計器へ追加する。"""
    for cls, mask in gts:
        map_state["gts"][cls].setdefault(image_key, []).append(mask)
    for cls, score, mask in preds:
        map_state["preds"][cls].append((image_key, float(score), mask))


def average_precision(preds, gts_by_image, iou_thr):
    """1 クラスの AP を IoU 閾値ごとに計算する。"""
    preds = sorted(preds, key=lambda x: -x[1])
    total_gts = sum(len(v) for v in gts_by_image.values())
    if total_gts == 0:
        return 0.0

    used = {key: np.zeros(len(v), dtype=bool) for key, v in gts_by_image.items()}
    tp = np.zeros(len(preds), dtype=float)
    fp = np.zeros(len(preds), dtype=float)
    for i, (image_key, _score, pmask) in enumerate(preds):
        gts = gts_by_image.get(image_key, [])
        flags = used.setdefault(image_key, np.zeros(len(gts), dtype=bool))
        best_iou, best_gt = iou_thr, -1
        for gi, gmask in enumerate(gts):
            if flags[gi]:
                continue
            iou = mask_iou(pmask, gmask)
            if iou >= best_iou:
                best_iou, best_gt = iou, gi
        if best_gt >= 0:
            flags[best_gt] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / total_gts
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return ap_from_pr(recalls, precisions)


def compute_map(map_state, n_cls, iou_thresholds=None):
    """mAP@0.5 と mAP@0.5:0.95 を計算する。"""
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
    ap_by_thr = {thr: {} for thr in iou_thresholds}
    for c in range(n_cls):
        preds = map_state["preds"][c]
        gts = map_state["gts"][c]
        for thr in iou_thresholds:
            ap_by_thr[thr][c] = average_precision(preds, gts, thr)

    ap50_by_class = {c: ap_by_thr[0.5][c] for c in range(n_cls)}
    ap5095_by_class = {
        c: float(np.mean([ap_by_thr[thr][c] for thr in iou_thresholds]))
        for c in range(n_cls)
    }
    return {
        "ap_by_thr": ap_by_thr,
        "ap50_by_class": ap50_by_class,
        "ap5095_by_class": ap5095_by_class,
        "map50": float(np.mean(list(ap50_by_class.values()))) if ap50_by_class else 0.0,
        "map5095": float(np.mean(list(ap5095_by_class.values()))) if ap5095_by_class else 0.0,
        "iou_thresholds": list(iou_thresholds),
    }


def match(preds, gts, iou_thr):
    """予測と正解を IoU 貪欲マッチング。(マッチ対, 未マッチpred, 未マッチgt) を返す。

    preds: [(cls, score, mask)] / gts: [(cls, mask)]
    マッチ対は (pred_idx, gt_idx, iou)。

    マッチングは予測スコア降順で行い、同一クラスの未使用 GT の中から
    IoU が最大かつ iou_thr 以上の 1 件だけに割り当てる。
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
            if pred_cls != gt_cls:
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
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics[c] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "iou": float(np.mean(ious)) if ious else 0.0,
            "iou_matches": len(ious),
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


def report(cm, metrics, labels, all_ious, map_metrics=None):
    """混同行列とクラス別指標（precision/recall/IoU）を表示。"""
    names = labels + ["background"]
    w = max(len(n) for n in names) + 2
    print("\n混同行列（行=正解 / 列=予測, 単位=インスタンス数）")
    print(" " * w + "".join(f"{n:>14}" for n in names))
    for i, row in enumerate(cm):
        print(f"{names[i]:<{w}}" + "".join(f"{int(v):>14}" for v in row))

        print("\nクラス別指標（rate と TP/FP/FN の生カウント）")
        print(f"{'class':<{w}}{'precision':>11}{'recall':>10}{'F1':>9}{'IoU':>9}"
            f"{'IoU n':>8}{'TP':>7}{'FP':>7}{'FN':>7}")
        for c, m in metrics.items():
          print(f"{labels[c]:<{w}}{m['precision']:>11.3f}{m['recall']:>10.3f}"
              f"{m['f1']:>9.3f}{m['iou']:>9.3f}{m['iou_matches']:>8}{m['tp']:>7}"
              f"{m['fp']:>7}{m['fn']:>7}")
        mp = np.mean([m["precision"] for m in metrics.values()])
        mr = np.mean([m["recall"] for m in metrics.values()])
        mf = np.mean([m["f1"] for m in metrics.values()])
        mi = np.mean([m["iou"] for m in metrics.values()])
        total_tp = sum(m["tp"] for m in metrics.values())
        total_fp = sum(m["fp"] for m in metrics.values())
        total_fn = sum(m["fn"] for m in metrics.values())
        micro_f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) else 0.0
        print(f"{'mean':<{w}}{mp:>11.3f}{mr:>10.3f}{mf:>9.3f}{mi:>9.3f}")
        print(f"{'micro':<{w}}{mp:>11.3f}{mr:>10.3f}{micro_f1:>9.3f}{mi:>9.3f}")
    if all_ious:
        print(f"\n全マッチインスタンスの平均 mask IoU: {np.mean(all_ious):.3f}  "
              f"(マッチ数 {len(all_ious)})")
        if map_metrics:
          print(f"mAP@0.5: {map_metrics['map50']:.3f}  mAP@0.5:0.95: {map_metrics['map5095']:.3f}")


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
    """画像群を評価し (混同行列, クラス別IoUリスト, 全IoUリスト, mAP状態) を返す。"""
    from PIL import Image

    bg = n_cls
    cm = np.zeros((n_cls + 1, n_cls + 1), dtype=np.int64)
    iou_by_class = {c: [] for c in range(n_cls)}
    all_ious = []
    map_state = init_map_state(n_cls)
    for path in paths:
        with Image.open(path) as im:
            w, h = im.size
        stem = os.path.splitext(os.path.basename(path))[0]
        gts = gt_masks(os.path.join(label_dir, f"{stem}.txt"), w, h)
        preds_all = predict_instances(model, path, min(score_thr, 0.001), (w, h))
        preds = [p for p in preds_all if p[1] >= score_thr]
        update_map_state(map_state, stem, preds_all, gts)

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
    return cm, iou_by_class, all_ious, map_state


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

    cm, iou_by_class, all_ious, map_state = evaluate_images(
        model, paths, label_dir, n_cls, iou_thr, score_thr)
    metrics = compute_metrics(cm, iou_by_class, n_cls)
    map_metrics = compute_map(map_state, n_cls)

    print(f"\n=== 評価: {weights} ===")
    print(f"画像 {len(paths)} 枚  IoU閾値={iou_thr}  スコア閾値={score_thr}")
    report(cm, metrics, labels, all_ious, map_metrics)
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
    p.add_argument(
        "--score-thrs",
        nargs="+",
        default=None,
        help="複数のスコア閾値を指定（例: --score-thrs 0.25 0.5 または 0.25,0.5）。"
             "指定時は --score-thr より優先。",
    )
    args = p.parse_args()
    if args.score_thrs:
        score_thrs = []
        for token in args.score_thrs:
            for raw in token.split(","):
                s = raw.strip()
                if not s:
                    raise SystemExit(
                        f"--score-thrs に空の要素があります: {args.score_thrs}"
                    )
                try:
                    score_thrs.append(float(s))
                except ValueError as e:
                    raise SystemExit(
                        f"--score-thrs の値が不正です: '{s}' (入力: {args.score_thrs})"
                    ) from e
    else:
        score_thrs = [args.score_thr]

    for score_thr in score_thrs:
        out_dir = args.output_dir
        if len(score_thrs) > 1:
            tag = f"{score_thr:.3f}".replace(".", "p")
            out_dir = os.path.join(args.output_dir, f"score_thr_{tag}")
        run(args.weights, args.images, args.labels, out_dir, args.iou_thr, score_thr)


if __name__ == "__main__":
    main()
