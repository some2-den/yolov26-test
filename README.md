# YOLO26-seg 建物損傷インスタンスセグメンテーション

DJI ドローン空撮画像から建物の損傷を検出する、YOLO26 セグメンテーション
（[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26)）の学習一式。

データは元々 YOLO seg 形式でアノテーションされているため、YOLO26 と相性が良い。

## クラス

| ID | ラベル |
|----|--------|
| 0  | No damage |
| 1  | Minor damage |
| 2  | Major damage |
| 3  | Blue sheet |

## 構成

| ファイル | 役割 |
|----------|------|
| `augment.py`     | 学習データのオフライン水増し（90/180/270 度回転） |
| `prepare_data.py`| YOLO 形式のデータセット構造（symlink）と data.yaml を生成 |
| `train.py`       | YOLO26-seg の学習 |
| `evaluate.py`    | 混同行列・クラス別 precision/recall/IoU の評価 |
| `infer.py`       | 推論・可視化（正解との比較画像 + 混同行列 + IoU） |

## データ配置

学習データは Mask2Former 版と共用（`~/asset/train_aug`、回転水増し済み 40 枚）。
`prepare_data.py` が以下の YOLO 規約の構造を symlink で組み立てる:

```
~/yolov26/dataset/images/train -> ~/asset/train_aug/images
~/yolov26/dataset/labels/train -> ~/asset/train_aug/labels
~/yolov26/dataset/images/val   -> ~/asset/validate
~/yolov26/dataset/labels/val   -> ~/asset/validate/data/labels/train
```

## 使い方

```bash
conda activate rf-detr
cd ~/yolov26

python augment.py     # 学習データの回転水増し（未生成なら）
python train.py       # YOLO26-seg を学習（終了時に自動評価）
python evaluate.py    # 混同行列・IoU・指標
python infer.py       # 比較画像 + 混同行列 + IoU
```

主なオプション:

```bash
python train.py --model yolo26s-seg.pt   # モデルサイズ変更（n/s/m/l/x）
python train.py --epochs 150 --batch 16
python infer.py --threshold 0.7          # スコア閾値
python evaluate.py --score-thrs 0.25 0.5 # 複数閾値を連続評価
```

成果物:
- `outputs/train/weights/best.pt` … 学習済みモデル
- `outputs/train/` … Ultralytics 標準の結果（学習曲線・混同行列など）
- `outputs/confusion_matrix.png` / `.csv` … 評価の混同行列
- `outputs/predictions/` … 正解との比較画像

## 注意点

- **元データが非常に少ない**（学習 10 / 検証 5 枚）。`augment.py` で回転水増し
  しているが、根本的な精度向上にはオリジナル画像の追加が必要。
- データ拡張は `augment.py` のオフライン回転に加え、Ultralytics 標準の
  オンライン拡張（mosaic・HSV・スケール・反転等）を併用する。mosaic は
  YOLO が性能を出すための中核機構のため、少数データでも有効化している。
- YOLO26 は NMS フリーの end-to-end 推論。データが YOLO seg 形式そのもの
  なので、形式変換のロスがない点が他手法に対する利点。
