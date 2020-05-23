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
           --input-size 512 \
           --normalizes-bboxes True \
           --num-runs 10 \
           --num-anchors-ratios 3 \
           --max-iter 300 \
           --min-size 0 \
           --iou-threshold 0.5 \
           --anchors-sizes 32, 64, 128, 256, 512 \
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
    --annotations-path annotations/instances_train2017.json \
    --input-size 512 \
    --normalizes-bboxes True \
    --num-runs 10 \
    --num-anchors-ratios 3 \
    --max-iter 300 \
    --min-size 0 \
    --iou-threshold 0.5 \
    --anchors-sizes 32 64 128 256 512 \
    --decimals 1
```
output:
```
2020-05-23 16:10:03 Starting the calculation of the optimal anchors ratios
2020-05-23 16:10:03 Reading annotations from annotations/instances_train2017.json
2020-05-23 16:10:23 Extracting and preprocessing bounding boxes
2020-05-23 16:10:28 Discarding 2 bounding boxes with size lower or equal to 0
2020-05-23 16:12:08 Best run avg. IoU: 80.48%
Runs avg. IoU: 80.48% ± 0.00% (mean ± std. dev. of 10 runs, 0 skipped)
2020-05-23 16:12:08 Avg. IoU between norm. anchors and bboxes: 80.48%
2020-05-23 16:12:08 Avg. IoU between bounding boxes and their most similar anchor: 55.72%
2020-05-23 16:12:15 Number of bounding boxes without similar anchors (IoU < 0.5):  240788/859999 (28.00%)
2020-05-23 16:12:15 Optimal anchors ratios: [(0.6, 1.5), (1.0, 1.0), (1.4, 0.7)]
2020-05-23 16:10:28 K-Means (10 runs): 100%|████████████| 10/10 [01:39<00:00,  9.94s/it]
```

# Updates

* [2020-05-23] add avg. IoU between bounding boxes and anchors
* [2020-05-23] add function to get annotations with bounding boxes without "similar" anchors
* [2020-05-23] add function to get anchors given ratios and sizes
* [2020-05-17] create this repository

## Acknowledgement

The code of this repo is mainly an adaptation of https://github.com/zhouyuangan/K-Means-Anchors.

Compared to the original, this code is faster and allows some additional settings.
