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
        "K-Means anchors rations calculator.",
        usage="python kmeans_anchors_ratios.py \\\n"
        "           --annotations-path path/to/your_coco_annotations.json \\\n"
        "           --input-size 512 \\\n"
        "           --normalizes-bboxes True \\\n"
        "           --num-runs 10 \\\n"
        "           --num-anchors-ratios 3 \\\n"
        "           --max-iter 300 \\\n"
        "           --min-size 0 \\\n"
        "           --iou-threshold 0.5 \\\n"
        "           --anchors-sizes 32 64 128 256 512 \\\n"
        "           --decimals 1",
    )
    parser.add_argument(
        "--annotations-path",
        type=str,
        metavar="N",
        help="Path to the json annotation file in COCO format.",
        required=True,
    )
    parser.add_argument(
        "--input-size",
        type=int,
        metavar="N",
        help="Size to which each image is scaled before being processed by the model.",
        required=True,
    )
    parser.add_argument(
        "--normalizes-bboxes",
        type=lambda x: True if x.strip().lower() == "true" else False,
        default=True,
        metavar="N",
        help="Normalizes bounding boxes, before giving them input to K-Means, so that "
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
        "--num-anchors-ratios",
        type=int,
        default=3,
        metavar="N",
        help="The number of anchors_ratios to generate. Default: 3.",
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
        help="Size to which all bounding box sizes must be stricly greater to be "
        "considered by K-Means. Filtering is applied after rescaling the bounding "
        "boxes to the same extent that the images are scaled to adapt them to the "
        "input size. Default: 0.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        metavar="N",
        help="Threshold above which anchors are assigned to ground-truth object boxes. "
        "Default: 0.5.",
    )
    parser.add_argument(
        "--anchors-sizes",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256, 512],
        metavar="N",
        help="List of anchors sizes. Default: [32, 64, 128, 256, 512].",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=1,
        metavar="N",
        help="Number of decimals to use when rounding anchors ratios. Default: 1.",
    )
    return parser.parse_known_args()


def iou(boxes, anchors):
    """Calculates the Intersection over Union (IoU) between a numpy array of
    n boxes and an array of k anchors.

    Arguments:
        boxes (numpy.ndarray): array of shape (n, 2) of boxes' widths and heights.
        anchors (numpy.ndarray): array of shape (k, 2) of anchors' widths and heights.

    Returns:
        A numpy array of shape (n, k) containing the IoU values ​​for each combination
        of boxes and anchors.
    """
    intersection_width = np.minimum(anchors[:, [0]], boxes[:, 0]).T
    intersection_height = np.minimum(anchors[:, [1]], boxes[:, 1]).T

    if np.any(intersection_width == 0) or np.any(intersection_height == 0):
        raise ValueError("Some boxes have zero size.")

    intersection_area = intersection_width * intersection_height
    boxes_area = np.prod(boxes, axis=1, keepdims=True)
    anchors_area = np.prod(anchors, axis=1, keepdims=True).T
    union_area = boxes_area + anchors_area - intersection_area
    return intersection_area / union_area


def avg_iou(boxes, anchors):
    """Calculates the average Intersection over Union (IoU) between a numpy array of
    boxes and k anchors.

    Arguments:
        boxes (numpy.ndarray): array of shape (n, 2) of boxes' widths and heights.
        anchors (numpy.ndarray): array of shape (k, 2) of anchors' widths and heights.

    Returns:
        The average of the IoU between the boxes and their nearest anchors.
    """
    return np.mean(np.max(iou(boxes, anchors), axis=1))


def kmeans(boxes, num_clusters=3, max_iter=300, seed=None, centroid_calc_fn=np.median):
    """Calculates K-Means clustering using the Intersection over Union (IoU) metric.

    Arguments:
        boxes (numpy.ndarray): array of the bounding boxes' heights and widths, with
            shape (n, 2).
        num_clusters (int): the number of clusters to form as well as the number of
            centroids to generate.
        max_iter (int, optional): maximum number of iterations of the K-Means algorithm
        for a single run. Default: 300.
        seed: (int, optional): random seed. Default: None.
        centroid_calc_fn (function, optional): function used for calculating centroids
            Default: numpy.median.

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


def get_anchors_from_ratios_and_sizes(anchors_ratios, anchors_sizes):
    anchors_ratios = np.asarray(anchors_ratios)
    anchors_sizes = np.asarray(anchors_sizes).reshape(-1, 1, 1)
    return (anchors_ratios.reshape(1, *anchors_ratios.shape) * anchors_sizes).reshape(
        -1, 2
    )


def get_bboxes_adapted_to_input_size(annotations, input_size):
    # scale factors used to resize the images to the size expected by the model
    scale_factors = {
        ann["id"]: input_size / max(ann["width"], ann["height"])
        for ann in annotations["images"]
    }
    return np.array(
        [
            np.array(ann["bbox"][-2:]) * scale_factors[ann["image_id"]]
            for ann in annotations["annotations"]
        ]
    )


def get_annotations_without_similar_anchors(
    annotations,
    anchors_ratios,
    anchors_sizes,
    input_size,
    iou_threshold=0.5,
    min_size=0,
):
    # get bboxes adapted to the input size
    bboxes = get_bboxes_adapted_to_input_size(annotations, input_size)
    # filter if size < min size
    have_size_gr_min_size = np.prod(bboxes, axis=1) > min_size
    bboxes = bboxes[have_size_gr_min_size]
    annotations = annotations["annotations"]
    annotations = [ann for ann, cond in zip(annotations, have_size_gr_min_size) if cond]
    # get anchors
    anchors = get_anchors_from_ratios_and_sizes(anchors_ratios, anchors_sizes)
    return [
        ann
        for ann, cond in zip(
            annotations, np.max(iou(bboxes, anchors), axis=1) < iou_threshold
        )
        if cond
    ]


def get_optimal_anchors_ratios(
    annotations_path,
    input_size,
    normalizes_bboxes=True,
    num_runs=1,
    num_anchors_ratios=3,
    max_iter=300,
    min_size=0,
    iou_threshold=0.5,
    anchors_sizes=[32, 64, 128, 256, 512],
    decimals=1,
):
    """Get the optimal anchors ratios using K-Means.

    Arguments:
        annotations_path (str): path to the json annotation file in COCO format.
        input_size (int): size to which each image is scaled before being processed by
            the model.
        normalizes_bboxes (bool, optional) normalizes bounding boxes, before giving them
            input to K-Means, so that they have all an area of ​​1. Default: True.
        num_runs (int, optional) how many times to run K-Means. After the end of all
            runs the best result is returned. Default: 1.
        num_anchors_ratios (int, optional): the number of anchors_ratios to generate
            Default: 3.
        max_iter (int, optional): maximum number of iterations of the K-Means algorithm
        for a single run. Default: 300.
        min_size (int, optional): size to which all bounding boxes must be stricly
            greater to. Filtering is applied after rescaling the bounding boxes to the
            same extent that the images are scaled to adapt them to the input size.
            Default: 0.
        iou_threshold (float, optional): Threshold above which anchors are assigned to
            ground-truth object boxes. Default: 0.5.
        anchors_sizes (list, optional): List of anchors sizes. Default: [32, 64, 128,
            256, 512].
        decimals (int, optional) . number of decimals to use when rounding anchors
            ratios. Default: 1.

    Returns:
        anchors_ratios as a list of tuple.
    """
    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        "Starting the calculation of the optimal anchors ratios"
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
    # resize the bounding boxes before filtering using images scale factors
    bboxes = get_bboxes_adapted_to_input_size(annotations, input_size)
    # filter the bounding boxes that are too small
    bboxes_ge_min_size = np.prod(bboxes, axis=1) > min_size
    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"Discarding {(~bboxes_ge_min_size).sum()} bounding boxes with size "
        f"lower or equal to {min_size}"
    )
    bboxes = bboxes[bboxes_ge_min_size]
    num_bboxes = len(bboxes)
    assert num_bboxes, "There is no bounding box left after filtering by size."

    if normalizes_bboxes:
        normalized_bboxes = bboxes / np.sqrt(bboxes.prod(axis=1, keepdims=True))
    else:
        normalized_bboxes = bboxes

    avg_iou_perc_list = []
    anchors_ratios_list = []
    pbar = tqdm(
        range(num_runs),
        desc=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"K-Means ({num_runs} run{'s' if num_runs > 1 else ''})",
        ncols=88,
    )
    for _ in pbar:
        ar = kmeans(
            normalized_bboxes, num_clusters=num_anchors_ratios, max_iter=max_iter
        )
        avg_iou_perc = avg_iou(normalized_bboxes, ar) * 100
        if np.isfinite(avg_iou_perc):
            anchors_ratios_list.append(ar)
            avg_iou_perc_list.append(avg_iou_perc)
        else:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"Skipping a run due to a numerical error in K-Means"
            )

    assert len(anchors_ratios_list), "No run was successful, try increasing num_runs."

    avg_iou_argmax = np.argmax(avg_iou_perc_list)
    # scaling to make the product of anchors ratios equal to 1
    anchors_ratios = anchors_ratios_list[avg_iou_argmax] / np.sqrt(
        anchors_ratios_list[avg_iou_argmax].prod(axis=1, keepdims=True)
    )
    # rounding of values ​​(only for aesthetic reasons)
    anchors_ratios = anchors_ratios.round(decimals)
    # get anchors
    anchors = get_anchors_from_ratios_and_sizes(anchors_ratios, anchors_sizes)
    # from array to list of tuple (standard format)
    anchors_ratios = sorted([tuple(ar) for ar in anchors_ratios])
    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"Best run avg. IoU: {avg_iou_perc_list[avg_iou_argmax]:.2f}%\n"
        f"Runs avg. IoU: {np.mean(avg_iou_perc_list):.2f}% ± "
        f"{np.std(avg_iou_perc_list):.2f}% "
        f"(mean ± std. dev. of {len(anchors_ratios_list)} runs, "
        f"{num_runs - len(anchors_ratios_list)} skipped)"
    )
    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"Avg. IoU between norm. anchors and bboxes: "
        f"{avg_iou_perc_list[avg_iou_argmax]:.2f}%"
    )
    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"Avg. IoU between bounding boxes and their most similar anchor: "
        f"{avg_iou(bboxes, anchors) * 100:.2f}%"
    )
    annotations = get_annotations_without_similar_anchors(
        annotations,
        anchors_ratios,
        anchors_sizes,
        input_size,
        iou_threshold=iou_threshold,
        min_size=min_size,
    )
    num_without_similar_anchors = len(annotations)
    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"Number of bounding boxes without similar anchors (IoU < {iou_threshold}): "
        f" {num_without_similar_anchors}/{num_bboxes} "
        f"({100 * num_without_similar_anchors / num_bboxes:.2f}%)"
    )
    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"Optimal anchors ratios: {anchors_ratios}"
    )
    return anchors_ratios


if __name__ == "__main__":
    args, _ = _parse_args()
    _ = get_optimal_anchors_ratios(**vars(args))
