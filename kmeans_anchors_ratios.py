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
        "K-Means anchors ratios calculator.",
        usage="python kmeans_anchors_ratios.py \\\n"
        "           --instances path/to/your_coco_instances.json \\\n"
        "           --anchors-sizes 32 64 128 256 512 \\\n"
        "           --input-size 512 \\\n"
        "           --normalizes-bboxes True \\\n"
        "           --num-runs 3 \\\n"
        "           --num-anchors-ratios 3 \\\n"
        "           --max-iter 300 \\\n"
        "           --min-size 0 \\\n"
        "           --iou-threshold 0.5 \\\n"
        "           --decimals 1 \\\n"
        "           --default-anchors-ratios '[(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)]'",
    )
    parser.add_argument(
        "--instances",
        type=str,
        metavar="N",
        help="Path to the json instances file in COCO format.",
        required=True,
    )
    parser.add_argument(
        "--anchors-sizes",
        nargs="+",
        type=int,
        metavar="N",
        help="List of anchors sizes (e.g. [32, 64, 128, 256, 512]).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        metavar="N",
        help="Size according to which each image is resized before being processed by "
        "the model.",
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
        help="The number of anchors ratios to generate. Default: 3.",
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
        "input_size. min_size=32 implies that that all the bounding boxes with an "
        "area less than or equal to 1024 (32 * 32) will be filtered. Default: 0.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        metavar="N",
        help="Threshold above which anchors are assigned to ground-truth object "
        "boxes. Default: 0.5.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=1,
        metavar="N",
        help="Number of decimals to use when rounding anchors ratios. Default: 1.",
    )
    parser.add_argument(
        "--default-anchors-ratios",
        type=eval,
        default=[(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)],
        metavar="N",
        help="List of anchors ratios to be compared with those found by K-Means. "
        "It must be passed as a string, e.g. '[(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)]'. "
        "Default: [(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)].",
    )
    return parser.parse_known_args()


def iou(boxes, anchors):
    """Calculates the Intersection over Union (IoU) between a numpy array of
    n boxes and an array of k anchors.

    Arguments:
        boxes (array_like): array of shape (n, 2) of boxes' widths and heights.
        anchors (array_like): array of shape (k, 2) of anchors' widths and heights.

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


def average_iou(boxes, anchors):
    """Calculates the average Intersection over Union (IoU) between a numpy array of
    boxes and k anchors.

    Arguments:
        boxes (array_like): array of shape (n, 2) of boxes' widths and heights.
        anchors (array_like): array of shape (k, 2) of anchors' widths and heights.

    Returns:
        The average of the IoU between the boxes and their nearest anchors.
    """
    return np.mean(np.max(iou(boxes, anchors), axis=1))


def kmeans(boxes, num_clusters=3, max_iter=300, seed=None, centroid_calc_fn=np.median):
    """Calculates K-Means clustering using the Intersection over Union (IoU) metric.

    Arguments:
        boxes (array_like): array of the bounding boxes' heights and widths, with
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
                f"[{datetime.now().strftime('%m/%d %H:%M:%S')}] "
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


def generate_anchors_given_ratios_and_sizes(anchors_ratios, anchors_sizes):
    """Generate anchors given anchors ratios and sizes.

    Arguments:
        anchors_ratios (array_like): array of anchors ratios (e.g. [(0.7, 1.4),
            (1.0, 1.0), (1.4, 0.7)]).
        anchors_sizes (array_like): array of anchors sizes (e.g. [32, 64, 128, 256,
            512]).

    Returns:
        An array of anchors.
    """
    anchors_ratios = np.asarray(anchors_ratios)
    anchors_sizes = np.asarray(anchors_sizes).reshape(-1, 1, 1)
    return (anchors_ratios.reshape(1, *anchors_ratios.shape) * anchors_sizes).reshape(
        -1, 2
    )


def get_bboxes_adapted_to_input_size(instances, input_size):
    """Extract the bounding boxes (widths and heights) from COCO-like annotations and
    adjust them to the input size.

    Arguments:
        instances (dict): COCO-like instances.
        input_size (int): size according to which each image is resized before being
            processed by the model.

    Returns:
        An array of resized bounding boxes.
    """
    # scale factors used to resize the images to the size expected by the model
    scale_factors = {
        ann["id"]: input_size / max(ann["width"], ann["height"])
        for ann in instances["images"]
    }
    return np.array(
        [
            np.array(ann["bbox"][-2:]) * scale_factors[ann["image_id"]]
            for ann in instances["annotations"]
        ]
    )


def get_annotations_without_similar_anchors(
    instances, anchors_ratios, anchors_sizes, input_size, iou_threshold=0.5, min_size=0
):
    """Get annotations whose bounding boxes don't have similar anchors.

    Arguments:
        instances (dict): COCO-like instances.
        anchors_ratios (array_like): array of anchors ratios (e.g. [(0.7, 1.4),
            (1.0, 1.0), (1.4, 0.7)]).
        anchors_sizes (array_like): array of anchors sizes (e.g. 32, 64, 128, 256,
            512).
        input_size (int): size according to which each image is resized before being
            processed by the model.
        iou_threshold (float, optional): Threshold above which anchors are assigned to
            ground-truth object boxes. Default: 0.5.
        min_size (int, optional): size to which all bounding boxes must be stricly
            greater to. Filtering is applied after rescaling the bounding boxes to the
            same extent that the images are scaled to adapt them to the input size.
            min_size=32 implies that that all the bounding boxes with an area less
            than or equal to 1024 (32 * 32) will be filtered. Default: 0.

    Returns:
        All the annotations in annotations["annotations"] whose bounding boxes don't
        have similar anchors.
    """
    # get bounding boxes adapted to the input size
    bboxes = get_bboxes_adapted_to_input_size(instances, input_size)
    # filter if size < min size
    have_size_gr_min_size = np.prod(bboxes, axis=1) > min_size ** 2
    bboxes = bboxes[have_size_gr_min_size]
    annotations = [
        ann
        for ann, cond in zip(instances["annotations"], have_size_gr_min_size)
        if cond
    ]
    # get anchors
    anchors = generate_anchors_given_ratios_and_sizes(anchors_ratios, anchors_sizes)
    return [
        ann
        for ann, cond in zip(
            annotations, np.max(iou(bboxes, anchors), axis=1) < iou_threshold
        )
        if cond
    ]


def get_optimal_anchors_ratios(
    instances,
    anchors_sizes,
    input_size,
    normalizes_bboxes=True,
    num_runs=1,
    num_anchors_ratios=3,
    max_iter=300,
    iou_threshold=0.5,
    min_size=0,
    decimals=1,
    default_anchors_ratios=[(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)],
):
    """Get the optimal anchors ratios using K-Means.

    Arguments:
        instances (dict): COCO-like instances.
        anchors_sizes (array_like): array of anchors sizes (e.g. 32, 64, 128, 256,
            512).
        input_size (int): size according to which each image is resized before being
            processed by the model.
        normalizes_bboxes (bool, optional) normalizes bounding boxes, before giving them
            input to K-Means, so that they have all an area of ​​1. Default: True.
        num_runs (int, optional) how many times to run K-Means. After the end of all
            runs the best result is returned. Default: 1.
        num_anchors_ratios (int, optional): the number of anchors_ratios to generate.
            Default: 3.
        max_iter (int, optional): maximum number of iterations of the K-Means algorithm
        for a single run. Default: 300.
        iou_threshold (float, optional): Threshold above which anchors are assigned to
            ground-truth object boxes. Default: 0.5.
        min_size (int, optional): size to which all bounding boxes must be stricly
            greater to. Filtering is applied after rescaling the bounding boxes to the
            same extent that the images are scaled to adapt them to the input size.
            min_size=32 implies that that all the bounding boxes with an area less
            than or equal to 1024 (32 * 32) will be filtered. Default: 0.
        decimals (int, optional): number of decimals to use when rounding anchors
            ratios. Default: 1.
        default_anchors_ratios (list, optional): list of anchors ratios to be compared
            with those found by K-Means. Default: [(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)].

    Returns:
        anchors_ratios as a list of tuple.
    """
    logger.info(
        f"[{datetime.now().strftime('%m/%d %H:%M:%S')}] "
        "Starting the calculation of the optimal anchors ratios"
    )
    logger.info(
        f"[{datetime.now().strftime('%m/%d %H:%M:%S')}] "
        "Extracting and preprocessing bounding boxes"
    )
    # resize the bounding boxes before filtering using images scale factors
    bboxes = get_bboxes_adapted_to_input_size(instances, input_size)
    # filter the bounding boxes that are too small
    bboxes_ge_min_size = np.prod(bboxes, axis=1) > min_size ** 2
    logger.info(
        f"[{datetime.now().strftime('%m/%d %H:%M:%S')}] "
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
        desc=f"[{datetime.now().strftime('%m/%d %H:%M:%S')}] "
        f"K-Means ({num_runs} run{'s' if num_runs > 1 else ''})",
        ncols=88,
    )
    for _ in pbar:
        ar = kmeans(
            normalized_bboxes, num_clusters=num_anchors_ratios, max_iter=max_iter
        )
        avg_iou_perc = average_iou(normalized_bboxes, ar) * 100
        if np.isfinite(avg_iou_perc):
            anchors_ratios_list.append(ar)
            avg_iou_perc_list.append(avg_iou_perc)
        else:
            logger.info(
                f"[{datetime.now().strftime('%m/%d %H:%M:%S')}] "
                f"Skipping a run due to numerical errors in K-Means"
            )

    assert len(anchors_ratios_list), "No run was successful, try increasing num_runs."

    avg_iou_argmax = np.argmax(avg_iou_perc_list)
    # scaling to make the product of anchors ratios equal to 1
    anchors_ratios = anchors_ratios_list[avg_iou_argmax] / np.sqrt(
        anchors_ratios_list[avg_iou_argmax].prod(axis=1, keepdims=True)
    )
    # rounding of values ​​(only for aesthetic reasons)
    anchors_ratios = anchors_ratios.round(decimals)
    # from array to list of tuple (standard format)
    anchors_ratios = sorted([tuple(ar) for ar in anchors_ratios])
    logger.info(
        f"\tRuns avg. IoU: {np.mean(avg_iou_perc_list):.2f}% ± "
        f"{np.std(avg_iou_perc_list):.2f}% "
        f"(mean ± std. dev. of {len(anchors_ratios_list)} runs, "
        f"{num_runs - len(anchors_ratios_list)} skipped)"
    )
    logger.info(
        "\tAvg. IoU between bboxes and their most similar anchors after "
        "norm. them to make their area equal (only ratios matter): "
        f"{avg_iou_perc_list[avg_iou_argmax]:.2f}%"
    )
    num_anns = len(instances["annotations"])
    # get default anchors
    default_anchors = generate_anchors_given_ratios_and_sizes(
        default_anchors_ratios, anchors_sizes
    )
    # get annotations without similar default anchors
    default_annotations = get_annotations_without_similar_anchors(
        instances,
        default_anchors_ratios,
        anchors_sizes,
        input_size,
        iou_threshold=iou_threshold,
        min_size=min_size,
    )
    num_bboxes_without_similar_default_anchors = len(default_annotations)
    default_perc_without = 100 * num_bboxes_without_similar_default_anchors / num_anns
    logger.info(
        f"[{datetime.now().strftime('%m/%d %H:%M:%S')}] "
        f"Default anchors ratios: {default_anchors_ratios}"
    )
    logger.info(
        f"\tAvg. IoU between bboxes and their most similar default anchors, "
        "no norm. (both ratios and sizes matter): "
        f"{average_iou(bboxes, default_anchors) * 100:.2f}%"
    )
    logger.info(
        f"\tNum. bboxes without similar default anchors (IoU < {iou_threshold}): "
        f" {num_bboxes_without_similar_default_anchors}/{num_anns} "
        f"({default_perc_without:.2f}%)"
    )
    # get K-Means anchors
    anchors = generate_anchors_given_ratios_and_sizes(anchors_ratios, anchors_sizes)
    # get annotations without similar K-Means anchors
    annotations = get_annotations_without_similar_anchors(
        instances,
        anchors_ratios,
        anchors_sizes,
        input_size,
        iou_threshold=iou_threshold,
        min_size=min_size,
    )
    num_bboxes_without_similar_anchors = len(annotations)
    perc_without = 100 * num_bboxes_without_similar_anchors / num_anns
    logger.info(
        f"[{datetime.now().strftime('%m/%d %H:%M:%S')}] "
        f"K-Means anchors ratios: {anchors_ratios}"
    )
    logger.info(
        f"\tAvg. IoU between bboxes and their most similar K-Means anchors, "
        "no norm. (both ratios and sizes matter): "
        f"{average_iou(bboxes, anchors) * 100:.2f}%"
    )
    logger.info(
        f"\tNum. bboxes without similar K-Means anchors (IoU < {iou_threshold}): "
        f" {num_bboxes_without_similar_anchors}/{num_anns} "
        f"({perc_without:.2f}%)"
    )
    if default_perc_without > perc_without:
        logger.info(
            f"[{datetime.now().strftime('%m/%d %H:%M:%S')}] "
            f"K-Means anchors have an IoU < {100 * iou_threshold:.0f}% with bboxes in "
            f"{default_perc_without - perc_without:.2f}% less cases than the default "
            "anchors, you should consider to use them"
        )
    else:
        logger.info(
            f"[{datetime.now().strftime('%m/%d %H:%M:%S')}] "
            f"Default anchors have an IoU < {100 * iou_threshold:.0f}% with bboxes in "
            f"{perc_without - default_perc_without:.2f}% less cases than the K-Means "
            "anchors, you should consider stick with them"
        )
    return anchors_ratios


if __name__ == "__main__":
    args, _ = _parse_args()
    logger.info(
        f"[{datetime.now().strftime('%m/%d %H:%M:%S')}] Reading {args.instances}"
    )
    with open(args.instances) as f:
        args = vars(args)
        args["instances"] = json.load(f)

    _ = get_optimal_anchors_ratios(**args)
