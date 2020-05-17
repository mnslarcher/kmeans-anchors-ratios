# K-Means Anchor Ratios Calculator

## Example of usage

```bash
$ pip install -r requirements.txt
$ python kmeans_anchor_ratios.py --help
```
output:
```
usage: python kmeans_anchor_ratios.py \
           --annotations-path path/to/your_coco_annotations.json \
           --input-size 512 \
           --scale-bboxes True \
           --num-runs 10 \
           --num-anchor-ratios 3 \
           --max-iter 300 \
           --min-size 0

optional arguments:
  -h, --help            show this help message and exit
  --annotations-path N  Path to the json annotation file in COCO format.
  --input-size N        Size to which each image is scaled before being
                        processed by the model.
  --scale-bboxes N      scale the bounding boxes, before giving them input to
                        K-Means, so that they have all an area of ​​1.
                        Default: True.
  --num-runs N          How many times to run K-Means. After the end of all
                        runs the best result is returned. Default: 1.
  --num-anchor-ratios N
                        The number of anchor_ratios to generate. Default: 3.
  --max-iter N          Maximum number of iterations of the K-Means algorithm
                        for a single run. Default: 300.
  --min-size N          Size to which all bounding box sizes must be stricly
                        greater to be considered by K-Means. Filtering is
                        applied after rescaling the bounding boxes to the same
                        extent that the images are scaled to adapt them to the
                        input size. Default: 0.
```
```bash
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ unzip annotations_trainval2017.zip
$ python kmeans_anchor_ratios.py \
       --annotations-path ./annotations/instances_train2017.json \
       --input-size 512 \
       --scale-bboxes True \
       --num-runs 10 \
       --num-anchor-ratios 3 \
       --max-iter 300 \
       --min-size 0
```
output:
```
2020-05-17 17:30:09 Starting the calculation of the optimal anchor ratios
2020-05-17 17:30:09 Reading annotations from ./annotations/instances_train2017.json
2020-05-17 17:30:25 Extracting and preprocessing bounding boxes
2020-05-17 17:30:36 Discarding 2 bounding boxes with size lower or equal to 0
2020-05-17 17:30:37 K-Means (10 runs): 100%|████████████| 10/10 [01:10<00:00,  7.05s/it]
2020-05-17 17:31:47 Runs avg. IoU: 80.48% ± 0.00% (mean ± std. dev. of 10 runs, 0 skipped)
2020-05-17 17:31:47 Optimal anchor ratios (avg. IoU: 80.48%): [(0.6, 1.5), (1.0, 1.0), (1.4, 0.7)]
```

## Acknowledgement

The code of this repo is mainly an adaptation of https://github.com/zhouyuangan/K-Means-Anchors.

Compared to the original, this code is faster and allows some additional settings.
