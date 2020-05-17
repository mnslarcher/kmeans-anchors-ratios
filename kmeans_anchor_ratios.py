import argparse
from datetime import datetime
import json
import logging
import numpy as np
import sys
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _parse_args():
    parser = argparse.ArgumentParser(
        "K-Means anchor rations calculator.",
        usage="python kmeans_anchor_ratios.py \\\n"
        "           --annotations-path path/to/your_coco_annotations.json \\\n"
        "           --input-size 512 \\\n"
        "           --scale-bboxes True \\\n"
        "           --num-runs 10 \\\n"
        "           --num-anchor-ratios 3 \\\n"
        "           --max-iter 300 \\\n"
        "           --min-size 0",
    )
    parser.add_argument(
        "--annotations-path",
        type=str,
        metavar="N",
        help="Path to the json annotation file in COCO format.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        metavar="N",
        help="Size to which each image is scaled before being processed by the model.",
    )
    parser.add_argument(
        "--scale-bboxes",
        type=lambda x: True if x.strip().lower() == "true" else False,
        default=True,
        metavar="N",
        help="scale the bounding boxes, before giving them input to K-Means, so that "
        "they have all an area of ​​1. Default: True.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        metavar="N",
        help="How many times to run K-Means. After the end of all runs the best result "
        "is returned. Default: 1.",
    )
    parser.add_argument(
        "--num-anchor-ratios",
        type=int,
        default=3,
        metavar="N",
        help="The number of anchor_ratios to generate. Default: 3.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        metavar="N",
        help="Maximum number of iterations of the K-Means algorithm for a single run. "
        "Default: 300.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=0,
        metavar="N",
        help=" Size to which all bounding box sizes must be stricly greater to be "
        "considered by K-Means. Filtering is applied after rescaling the bounding "
        "boxes to the same extent that the images are scaled to adapt them to the "
        "input size. Default: 0.",
    )
    return parser.parse_known_args()


def iou(boxes, centroids):
    """Calculates the Intersection over Union (IoU) between a numpy array of
    n boxes and an array of k centroids.

    Arguments:
        boxes (numpy.ndarray): array of shape (n, 2) of boxes' widths and heights.
        centroids (numpy.ndarray): array of shape (k, 2) of centroids, where k is the
            number of clusters.

    Returns:
        A numpy array of shape (n, k) containing the IoU values ​​for each combination
        of boxes and centroids.
    """
    intersection_width = np.minimum(centroids[:, [0]], boxes[:, 0]).T
    intersection_height = np.minimum(centroids[:, [1]], boxes[:, 1]).T

    if np.any(intersection_width == 0) or np.any(intersection_height == 0):
        raise ValueError("Some boxes have zero size.")

    intersection_area = intersection_width * intersection_height
    boxes_area = np.prod(boxes, axis=1, keepdims=True)
    centroids_area = np.prod(centroids, axis=1, keepdims=True).T
    union_area = boxes_area + centroids_area - intersection_area

    return intersection_area / (union_area)


def avg_iou(boxes, centroids):
    """Calculates the average Intersection over Union (IoU) between a numpy array of
    boxes and k centroids.

    Arguments:
        boxes (numpy.ndarray): array of shape (n, 2) of boxes' widths and heights.
        centroids (numpy.ndarray): array of shape (k, 2) of centroids, where k is the
            number of clusters.

    Returns:
        The average of the IoU between the boxes and their nearest centroids.
    """
    return np.mean(np.max(iou(boxes, centroids), axis=1))


def kmeans(boxes, num_clusters=3, max_iter=300, seed=None, centroid_calc_fn=np.median):
    """Calculates K-Means clustering using the Intersection over Union (IoU) metric.

    Arguments:
        boxes (numpy.ndarray): array of the bounding boxes' heights and widths, with
            shape (n, 2).
        num_clusters (int): the number of clusters to form as well as the number of
            centroids to generate.
        max_iter (int, optional): maximum number of iterations of the K-Means algorithm
        for a single run (default: 300).
        centroid_calc_fn (function, optional): function used for calculating centroids
            (default: numpy.median).

    Returns:
        A numpy array of shape (num_clusters, 2).
    """
    np.random.seed(seed)
    last_nearest_centroids = np.ones(boxes.shape[0]) * -1
    # the Forgy method will fail if the whole array contains the same rows
    centroids = boxes[np.random.choice(boxes.shape[0], num_clusters, replace=False)]
    i = 0
    while True:
        if i >= max_iter:
            logger.warning(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                "Maximum number of iterations reached, increase max_inter to do more "
                f"iterations (max_inter = {max_iter})"
            )
            break

        nearest_centroids = np.argmax(iou(boxes, centroids), axis=1)

        for centroid in range(num_clusters):
            centroids[centroid] = centroid_calc_fn(
                boxes[nearest_centroids == centroid], axis=0
            )

        if (nearest_centroids == last_nearest_centroids).all():
            break

        last_nearest_centroids = nearest_centroids
        i += 1

    return centroids


def get_optimal_anchor_ratios(
    annotations_path,
    input_size,
    scale_bboxes=True,
    num_runs=1,
    num_anchor_ratios=3,
    max_iter=300,
    min_size=0,
    decimals=1,
):
    """Get the optimal anchor ratios using K-Means.

    Arguments:
        annotations_path (str): path to the json annotation file in COCO format.
        input_size (int): size to which each image is scaled before being processed by
            the model.
        scale_bboxes (bool, optional) scale the bounding boxes, before giving them
            input to K-Means, so that they have all an area of ​​1 (default: True).
        num_runs (int, optional) how many times to run K-Means. After the end of all
            runs the best result is returned (default: 1).
        num_anchor_ratios (int, optional): the number of anchor_ratios to generate
            (default: 3).
        max_iter (int, optional): maximum number of iterations of the K-Means algorithm
        for a single run (default: 300).
        min_size (int, optional): size to which all bounding box sizes must be stricly
            greater to be considered by K-Means. Filtering is applied after rescaling
            the bounding boxes to the same extent that the images are scaled to adapt
            them to the input size (default: 0).

    Returns:
        anchor_ratios as a list of tuple.
    """
    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        "Starting the calculation of the optimal anchor ratios"
    )
    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"Reading annotations from {annotations_path}"
    )
    with open(annotations_path) as f:
        annotations = json.load(f)

    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        "Extracting and preprocessing bounding boxes"
    )
    # ground truth bounding boxes
    bboxes = np.array(
        [
            ann["bbox"][-2:]
            for ann in annotations["annotations"]
            if np.prod(ann["bbox"][-2:]) > min_size
        ]
    )
    # scale factors used to resize the images to the size expected by the model
    scale_factors = {
        ann["id"]: input_size / max(ann["width"], ann["height"])
        for ann in annotations["images"]
    }
    # resize the bounding boxes before filtering using images scale factors
    bboxes = np.array(
        [
            np.array(ann["bbox"][-2:]) * scale_factors[ann["image_id"]]
            for ann in annotations["annotations"]
        ]
    )
    # filter the bounding boxes that are too small
    bboxes_gr_min_size = np.prod(bboxes, axis=1) > min_size
    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"Discarding {(~bboxes_gr_min_size).sum()} bounding boxes with size "
        f"lower or equal to {min_size}"
    )
    bboxes = bboxes[bboxes_gr_min_size]
    assert len(bboxes), "There is no bounding box left after filtering by size."

    if scale_bboxes:
        bboxes = bboxes / np.sqrt(bboxes.prod(axis=1, keepdims=True))

    avg_iou_perc_list = []
    centroids_list = []
    pbar = tqdm(
        range(num_runs),
        desc=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"K-Means ({num_runs} run{'s' if num_runs > 1 else ''})",
        ncols=88,
    )
    for _ in pbar:
        centroids = kmeans(bboxes, num_clusters=num_anchor_ratios, max_iter=max_iter)
        avg_iou_perc = avg_iou(bboxes, centroids) * 100
        if np.isfinite(avg_iou_perc):
            centroids_list.append(centroids)
            avg_iou_perc_list.append(avg_iou_perc)
        else:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"Skipping a run due to a numerical error in K-Means"
            )

    assert len(centroids_list), "No run was successful, try increasing num_runs."

    avg_iou_argmax = np.argmax(avg_iou_perc_list)
    # scaling to make the product of anchor ratios equal to 1
    anchor_ratios = centroids_list[avg_iou_argmax] / np.sqrt(
        centroids_list[avg_iou_argmax].prod(axis=1, keepdims=True)
    )
    # rounding of values ​​(only for aesthetic reasons)
    anchor_ratios = anchor_ratios.round(decimals)
    # from array to list of tuple (standard format)
    anchor_ratios = sorted([tuple(ar) for ar in anchor_ratios])
    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"Optimal anchor ratios (avg. IoU: {avg_iou_perc_list[avg_iou_argmax]:.2f}%): "
        f"{anchor_ratios}"
    )
    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"Runs avg. IoU: {np.mean(avg_iou_perc_list):.2f}% ± "
        f"{np.std(avg_iou_perc_list):.2f}% "
        f"(mean ± std. dev. of {len(centroids_list)} runs, "
        f"{num_runs - len(centroids_list)} skipped)"
    )
    return anchor_ratios


if __name__ == "__main__":
    args, _ = _parse_args()
    _ = get_optimal_anchor_ratios(**vars(args))
