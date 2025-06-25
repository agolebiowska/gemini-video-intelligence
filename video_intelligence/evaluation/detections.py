import argparse
import json
from collections import defaultdict
import numpy as np


class BoundingBox:
    """
    Represents a bounding box with coordinates [ymin, xmin, ymax, xmax] (PASCAL VOC format)
    and provides methods for IoU calculation.
    """

    def __init__(self, ymin, xmin, ymax, xmax):
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax

    def area(self):
        """Calculates the area of the bounding box."""
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def intersection(self, other):
        """Calculates the intersection area with another BoundingBox."""
        x_left = max(self.xmin, other.xmin)
        y_top = max(self.ymin, other.ymin)
        x_right = min(self.xmax, other.xmax)
        y_bottom = min(self.ymax, other.ymax)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        return (x_right - x_left) * (y_bottom - y_top)

    def iou(self, other):
        """Calculates the Intersection over Union (IoU) with another BoundingBox."""
        intersection_area = self.intersection(other)
        union_area = self.area() + other.area() - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0


class ObjectDetectionEvaluator:
    """
    Evaluates object detection performance using mAP (11-point interpolation).
    """

    def __init__(
        self,
        ground_truth_file: str,
        predictions_file: str,
        iou_threshold: float = 0.3,
    ):
        self.ground_truth_file = ground_truth_file
        self.predictions_file = predictions_file
        self.iou_threshold = iou_threshold

    def _parse_results_file(self, file_path: str) -> dict:
        with open(file_path, 'r') as f:
            return json.load(f)

    def _calculate_class_metrics(
        self, gt_boxes: list, pred_boxes: list
    ) -> dict:
        """
        Calculates all relevant metrics for a single class: AP, Precision, Recall, TP, FP, FN.
        Uses 11-point interpolation for AP.
        """
        if not gt_boxes:
            tp = 0
            fp = len(pred_boxes)
            fn = 0
            precision = 0.0 if fp > 0 else 1.0
            recall = 0.0 if fp > 0 else 1.0
            ap = 0.0 if fp > 0 else 1.0
            return {
                'ap': ap,
                'precision': precision,
                'recall': recall,
                'tp': tp,
                'fp': fp,
                'fn': fn,
            }

        if not pred_boxes:
            return {
                'ap': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': len(gt_boxes),
            }

        pred_boxes.sort(key=lambda x: x[0], reverse=True)

        num_gt = len(gt_boxes)
        gt_matched = [False] * num_gt

        tp_cumulative = np.zeros(len(pred_boxes))
        fp_cumulative = np.zeros(len(pred_boxes))

        for i, (confidence, pred_box) in enumerate(pred_boxes):
            best_iou = -1
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                iou = pred_box.iou(gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= self.iou_threshold:
                if not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                    tp_cumulative[i] = 1
                else:
                    fp_cumulative[i] = 1
            else:
                fp_cumulative[i] = 1

        tp_cumulative = np.cumsum(tp_cumulative)
        fp_cumulative = np.cumsum(fp_cumulative)

        recalls = tp_cumulative / num_gt
        precisions = tp_cumulative / (tp_cumulative + fp_cumulative)

        ap = 0.0
        for recall_level in np.linspace(0, 1, 11):
            try:
                precisions_at_recall = precisions[recalls >= recall_level]
                max_precision = np.max(precisions_at_recall)
            except ValueError:
                max_precision = 0.0
            ap += max_precision

        ap /= 11.0

        total_tp = int(tp_cumulative[-1])
        total_fp = int(fp_cumulative[-1])
        total_fn = num_gt - total_tp
        final_precision = (
            total_tp / (total_tp + total_fp)
            if (total_tp + total_fp) > 0
            else 0.0
        )
        final_recall = total_tp / num_gt if num_gt > 0 else 0.0

        return {
            'ap': ap,
            'precision': final_precision,
            'recall': final_recall,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
        }

    def run_evaluation(self) -> dict:
        print("Parsing ground truth...")
        ground_truth_data = self._parse_results_file(self.ground_truth_file)

        print("Parsing predictions...")
        predictions_data = self._parse_results_file(self.predictions_file)

        gts_by_class = defaultdict(list)
        preds_by_class = defaultdict(list)

        all_video_paths = set(ground_truth_data.keys()) | set(
            predictions_data.keys()
        )
        for video_path in all_video_paths:
            gt_frames = {
                f['timestamp']: f['objects']
                for f in ground_truth_data.get(video_path, [])
            }
            pred_frames = {
                f['timestamp']: f['objects']
                for f in predictions_data.get(video_path, [])
            }
            all_timestamps = set(gt_frames.keys()) | set(pred_frames.keys())
            for ts in all_timestamps:
                for gt_obj in gt_frames.get(ts, []):
                    gts_by_class[gt_obj['label']].append(
                        BoundingBox(*gt_obj['box_2d'])
                    )
                for pred_obj in pred_frames.get(ts, []):
                    preds_by_class[pred_obj['label']].append(
                        (
                            pred_obj['confidence'],
                            BoundingBox(*pred_obj['box_2d']),
                        )
                    )

        print("Calculating metrics...")
        metrics_per_class = {}
        all_labels = sorted(
            list(set(gts_by_class.keys()) | set(preds_by_class.keys()))
        )

        for label in all_labels:
            metrics_per_class[label] = self._calculate_class_metrics(
                gts_by_class[label], preds_by_class[label]
            )

        ap_values = [metrics['ap'] for metrics in metrics_per_class.values()]
        mAP = np.mean(ap_values) if ap_values else 0.0

        results = {
            'metrics_per_class': metrics_per_class,
            'mAP': mAP,
            'config': {'iou_threshold': self.iou_threshold},
        }
        return results

    @staticmethod
    def display_results(results: dict):
        print("\n" + "=" * 85)
        print("OBJECT DETECTION EVALUATION RESULTS (11-Point Interpolation)")
        print("=" * 85)

        iou_thresh = results['config']['iou_threshold']
        print(f"\nConfiguration:")
        print(f"  IoU Threshold: {iou_thresh}")

        print(f"\nPER-CLASS METRICS:")
        print("-" * 85)
        print(
            f"{'Label':<20} {'AP':<10} {'Precision':<10} {'Recall':<10} {'TP':<7} {'FP':<7} {'FN':<7}"
        )
        print("-" * 85)

        for label, metrics in results['metrics_per_class'].items():
            print(
                f"{label:<20} {metrics['ap']:.4f}     "
                f"{metrics['precision']:.3f}     "
                f"{metrics['recall']:.3f}     "
                f"{metrics['tp']:<7} {metrics['fp']:<7} {metrics['fn']:<7}"
            )

        print("-" * 85)
        print(f"\nMean Average Precision (mAP): {results['mAP']:.4f}")
        print("\n" + "=" * 85)

    @staticmethod
    def save_results(results: dict, output_file: str):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate video object detection performance using mAP (11-point interpolation).'
    )
    parser.add_argument(
        'ground_truth_file', help='Path to ground truth JSON file'
    )
    parser.add_argument(
        'predictions_file', help='Path to predictions JSON file'
    )
    parser.add_argument(
        '--iou_threshold',
        type=float,
        default=0.3,
        help='IoU threshold for a correct detection (default: 0.3)',
    )
    parser.add_argument(
        '--output', help='Path to save detailed results JSON file'
    )
    args = parser.parse_args()

    evaluator = ObjectDetectionEvaluator(
        ground_truth_file=args.ground_truth_file,
        predictions_file=args.predictions_file,
        iou_threshold=args.iou_threshold,
    )

    results = evaluator.run_evaluation()
    evaluator.display_results(results)

    if args.output:
        evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
