import argparse
import json
import os
import pandas as pd
from collections import defaultdict


def load_video_metadata(metadata_file: str) -> dict[str, dict[str, dict]]:
    """Loads video metadata from a CSV file."""
    metadata_df = pd.read_csv(metadata_file)
    video_metadata = {}
    for _, row in metadata_df.iterrows():
        video_filename = os.path.basename(row['Video'])
        video_name, _ = os.path.splitext(video_filename)
        video_metadata[video_name] = {'fps': row['FPS']}
    return video_metadata


def convert_label_studio_annotations(
    labels_file: str, video_metadata: dict[str, dict[str, dict]]
) -> dict[str, list[dict]]:
    """
    Converts Label Studio annotations exported as JSON-MIN
    """
    with open(labels_file, 'r') as f:
        label_studio_data = json.load(f)

    all_video_results = {}

    for video_annotation in label_studio_data:
        ls_video_path = video_annotation['video']
        path_parts = ls_video_path.split('/')
        video_name = path_parts[-2]
        output_key = "/".join(path_parts[-3:])

        metadata = video_metadata.get(video_name)
        if not metadata:
            print(
                f"Warning: No metadata found for video {video_name}. Skipping."
            )
            continue
        fps = metadata['fps']

        detections_by_timestamp = defaultdict(list)

        for tracked_object in video_annotation['box']:
            label_list = tracked_object.get('labels')
            if not label_list:
                continue

            label = label_list[0]

            for annotation in tracked_object['sequence']:
                if not annotation.get('enabled', False):
                    continue

                frame = annotation['frame']
                if frame == 0:
                    continue

                total_seconds = int((frame - 1) / fps)
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                timestamp_str = f"{minutes:02d}:{seconds:02d}"

                width = annotation.get('width', 0)
                height = annotation.get('height', 0)
                if width <= 0 or height <= 0:
                    continue

                x, y = annotation['x'], annotation['y']
                box_2d = [
                    int(y * 10),
                    int(x * 10),
                    int((y + height) * 10),
                    int((x + width) * 10),
                ]
                detection = {
                    'box_2d': box_2d,
                    'label': label,
                    'confidence': 1.0,
                }

                detections_by_timestamp[timestamp_str].append(detection)

        timestamp_list = []
        for ts, detections in sorted(detections_by_timestamp.items()):
            timestamp_list.append({'timestamp': ts, 'objects': detections})

        all_video_results[output_key] = timestamp_list

    return all_video_results


def main():
    parser = argparse.ArgumentParser(
        description='Convert Label Studio annotations (JSON-MIN) to internal detection format for easier evaluation.'
    )
    parser.add_argument(
        'labels_file', help='Path to Label Studio annotations JSON file'
    )
    parser.add_argument(
        'metadata_file', help='Path to video metadata CSV file'
    )
    parser.add_argument(
        'output_file', help='Path to save the converted annotations'
    )
    args = parser.parse_args()

    video_metadata = load_video_metadata(args.metadata_file)
    converted_data = convert_label_studio_annotations(
        args.labels_file, video_metadata
    )

    with open(args.output_file, 'w') as f:
        json.dump(converted_data, f, indent=4)

    print(f"Converted annotations saved to {args.output_file}")


if __name__ == '__main__':
    main()
