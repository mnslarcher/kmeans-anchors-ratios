# K-Means Anchors Ratios Calculator

## Example of usage

```
$ pip install -r requirements.txt
$ python kmeans_anchors_ratios.py --help
```
output:
```
usage: python kmeans_anchors_ratios.py \
           --instances path/to/your_coco_instances.json \
           --anchors-sizes 32 64 128 256 512 \
           --input-size 512 \
           --normalizes-bboxes True \
           --num-runs 3 \
           --num-anchors-ratios 3 \
           --max-iter 300 \
           --min-size 0 \
           --iou-threshold 0.5 \
           --decimals 1 \
           --default-anchors-ratios '[(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)]'

optional arguments:
  -h, --help            show this help message and exit
  --instances N         Path to the json instances file in COCO format.
  --anchors-sizes N [N ...]
                        List of anchors sizes (e.g. [32, 64, 128, 256, 512]).
  --input-size N        Size according to which each image is resized before
                        being processed by the model.
  --normalizes-bboxes N
                        Normalizes bounding boxes, before giving them input to
                        K-Means, so that they have all an area of ​​1.
                        Default: True.
  --num-runs N          How many times to run K-Means. After the end of all
                        runs the best result is returned. Default: 1.
  --num-anchors-ratios N
                        The number of anchors ratios to generate. Default: 3.
  --max-iter N          Maximum number of iterations of the K-Means algorithm
                        for a single run. Default: 300.
  --min-size N          Size to which all bounding box sizes must be stricly
                        greater to be considered by K-Means. Filtering is
                        applied after rescaling the bounding boxes to the same
                        extent that the images are scaled to adapt them to the
                        input_size. min_size=32 implies that that all the
                        bounding boxes with an area less than or equal to 1024
                        (32 * 32) will be filtered. Default: 0.
  --iou-threshold N     Threshold above which anchors are assigned to ground-
                        truth object boxes. Default: 0.5.
  --decimals N          Number of decimals to use when rounding anchors
                        ratios. Default: 1.
  --default-anchors-ratios N
                        List of anchors ratios to be compared with those found
                        by K-Means. It must be passed as a string, e.g.
                        '[(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)]'. Default:
                        [(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)].
```
```
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ unzip annotations_trainval2017.zip
$ python kmeans_anchors_ratios.py \
    --instances ./annotations/instances_train2017.json \
    --anchors-sizes 32 64 128 256 512 \
    --input-size 512 \
    --normalizes-bboxes True \
    --num-runs 3 \
    --num-anchors-ratios 3 \
    --max-iter 300 \
    --min-size 0 \
    --iou-threshold 0.5 \
    --decimals 1 \
    --default-anchors-ratios '[(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)]'
```
output:
```
[06/13 12:57:38] Reading ./annotations/instances_train2017.json
[06/13 12:57:54] Starting the calculation of the optimal anchors ratios
[06/13 12:57:54] Extracting and preprocessing bounding boxes
[06/13 12:57:57] Discarding 2 bounding boxes with size lower or equal to 0
[06/13 12:57:57] K-Means (3 runs): 100%|██████████████████| 3/3 [00:33<00:00, 11.06s/it]
        Runs avg. IoU: 80.48% ± 0.00% (mean ± std. dev. of 3 runs, 0 skipped)
        Avg. IoU between bboxes and their most similar anchors after norm. them to make their area equal (only ratios matter):
80.48%
[06/13 12:58:33] Default anchors ratios: [(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)]
        Avg. IoU between bboxes and their most similar default anchors, no norm. (both ratios and sizes matter): 55.16%
        Num. bboxes without similar default anchors (IoU < 0.5):  253049/860001 (29.42%)
[06/13 12:58:37] K-Means anchors ratios: [(0.6, 1.5), (1.0, 1.0), (1.4, 0.7)]
        Avg. IoU between bboxes and their most similar K-Means anchors, no norm. (both ratios and sizes matter): 55.72%
        Num. bboxes without similar K-Means anchors (IoU < 0.5):  240788/860001 (28.00%)
[06/13 12:58:37] K-Means anchors have an IoU < 50% with bboxes in 1.43% less cases than the default anchors, you should consider to use them
```
For more infos see the [tutorial](tutorial.ipynb).
## Updates

* [2020-06-13] added a comparision with the default anchors ratios, as suggested by @zylo117
* [2020-05-23] added avg. IoU between bounding boxes and anchors
* [2020-05-23] added a function to get annotations whose bounding boxes don't have similar anchors
* [2020-05-23] added a function to generate anchors given ratios and sizes
* [2020-05-17] created this repository

## Acknowledgement

The code of this repo is mainly an adaptation of https://github.com/zhouyuangan/K-Means-Anchors.

Compared to the original, the K-Means implementation of this repo is more than 50x faster and some additional features are provided.

