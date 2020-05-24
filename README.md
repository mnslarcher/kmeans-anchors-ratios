# K-Means Anchors Ratios Calculator

## Example of usage

```bash
$ pip install -r requirements.txt
$ python kmeans_anchors_ratios.py --help
```
output:
```
usage: python kmeans_anchors_ratios.py \
           --annotations-path path/to/your_coco_annotations.json \
           --anchors-sizes 32, 64, 128, 256, 512 \
           --input-size 512 \
           --normalizes-bboxes True \
           --num-runs 10 \
           --num-anchors-ratios 3 \
           --max-iter 300 \
           --min-size 0 \
           --iou-threshold 0.5 \
           --decimals 1

optional arguments:
  -h, --help            show this help message and exit
  --annotations-path N  Path to the json annotation file in COCO format.
  --input-size N        Size to which each image is scaled before being processed by the model.
  --normalizes-bboxes N
                        Normalizes bounding boxes, before giving them input to K-Means, so that they have all an area of ​​1.
                        Default: True.
  --num-runs N          How many times to run K-Means. After the end of all runs the best result is returned. Default: 1.
  --num-anchors-ratios N
                        The number of anchors_ratios to generate. Default: 3.
  --max-iter N          Maximum number of iterations of the K-Means algorithm for a single run. Default: 300.
  --min-size N          Size to which all bounding box sizes must be stricly greater to be considered by K-Means. Filtering is
                        applied after rescaling the bounding boxes to the same extent that the images are scaled to adapt them to the
                        input size. Default: 0.
  --iou-threshold N     Threshold above which anchors are assigned to ground-truth object boxes. Default: 0.5.
  --anchors-sizes N [N ...]
                        List of anchors sizes. Default: [32, 64, 128, 256, 512].
  --decimals N          Number of decimals to use when rounding anchors ratios. Default: 1.
```
```bash
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ unzip annotations_trainval2017.zip
$ python kmeans_anchors_ratios.py \
    --annotations-path path/to/your_coco_annotations.json \
    --input-size 512 \
    --normalizes-bboxes True \
    --num-runs 10 \
    --num-anchors-ratios 3 \
    --max-iter 300 \
    --min-size 0 \
    --iou-threshold 0.5 \
    --anchors-sizes 32, 64, 128, 256, 512 \
    --decimals 1
```
output:
```
[05/24 14:20:50] Starting the calculation of the optimal anchors ratios
[05/24 14:20:50] Extracting and preprocessing bounding boxes
[05/24 14:20:53] Discarding 2 bounding boxes with size lower or equal to 0
[05/24 14:20:53] K-Means (10 runs): 100%|███████████████| 10/10 [01:41<00:00, 10.17s/it]
[05/24 14:22:34] Runs avg. IoU: 80.48% ± 0.00% (mean ± std. dev. of 10 runs, 0 skipped)
[05/24 14:22:34] Avg. IoU between bboxes and their most similar anchors after normalizing them so that they have the same area (only the anchor ratios matter): 80.48%
[05/24 14:22:34] Avg. IoU between bboxes and their most similar anchors (no normalization, both anchor ratios and sizes matter): 61.02%
[05/24 14:22:40] Num. bboxes with similar anchors (IoU >= 0.5):  620506/860001 (72.15%)
[05/24 14:22:40] Optimal anchors ratios: [(0.6, 1.5), (1.0, 1.0), (1.4, 0.7)]
```
For more infos see the [tutorial](tutorial.ipynb).
## Updates

* [2020-05-23] added avg. IoU between bounding boxes and anchors
* [2020-05-23] added a function to get annotations whose bounding boxes don't have similar anchors
* [2020-05-23] added a function to generate anchors given ratios and sizes
* [2020-05-17] created this repository

## Acknowledgement

The code of this repo is mainly an adaptation of https://github.com/zhouyuangan/K-Means-Anchors.

Compared to the original, this code is faster and allows some additional settings.
