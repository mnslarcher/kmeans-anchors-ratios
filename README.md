# K-Means Anchor Ratios Calculator

## Example of usage

```
python kmeans_anchor_ratios.py \
    --annotations-path "path/to/your_coco_annotations.json" \
    --input-size 512 \
    --scale-bboxes True \
    --num-runs 10 \
    --num-anchor-ratios 3 \
    --max-iter 300 \
    --min-size 0
```
Run
```
python kmeans_anchor_ratios.py --help
```
for more infos.

## Acknowledgement

The code of this repo is mainly an adaptation of https://github.com/zhouyuangan/K-Means-Anchors.

Compared to the original, this code is faster and allows some additional settings.
