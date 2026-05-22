"""YOLO26 セグメンテーションモデルを建物損傷データで学習する。

データは元々 YOLO seg 形式。学習データは augment.py のオフライン回転水増し
（90/180/270 度）を使い、さらに Ultralytics 標準のオンラインデータ拡張
（mosaic・HSV・スケール・反転など）を併用する。mosaic は YOLO が性能を出す
ための中核機構のため、少数データでも有効化する。
"""
from __future__ import annotations

import argparse
import os

from ultralytics import YOLO

import prepare_data


def main():
    home = os.path.expanduser("~")
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26n-seg.pt",
                   help="ベースモデル（yolo26n/s/m/l/x-seg.pt）")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    # データセット構造（symlink）と data.yaml を用意
    print("データセットを準備中...")
    data_yaml = prepare_data.main()

    model = YOLO(args.model)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=f"{home}/yolov26/outputs",
        name="train",
        exist_ok=True,
        # データ拡張は Ultralytics 標準（mosaic・HSV・スケール・反転など）を使用。
        # 引数を指定しないことで既定の拡張がそのまま有効になる。
    )

    best = f"{home}/yolov26/outputs/train/weights/best.pt"
    print(f"\n学習完了。best モデル: {best}")

    # 検証データで混同行列・IoU・クラス別指標を出力
    import evaluate
    evaluate.run(
        weights=best,
        image_dir=f"{home}/asset/validate",
        label_dir=f"{home}/asset/validate/data/labels/train",
        output_dir=f"{home}/yolov26/outputs",
    )


if __name__ == "__main__":
    main()
