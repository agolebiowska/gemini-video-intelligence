import json
import argparse
import sys
from pathlib import Path

from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from video_intelligence.config import Config


class BigQueryExporter:
    """Handles exporting video intelligence data to BigQuery."""

    def __init__(self, config: Config):
        self.config = config
        self.client = bigquery.Client()
        self.table_id = f"{self.config.project_id}.{self.config.dataset_id}.{self.config.table_id}"

    @property
    def schema(self) -> list[bigquery.SchemaField]:
        """Define the BigQuery table schema."""
        return [
            bigquery.SchemaField("video_path", "STRING", mode="REQUIRED"),
            bigquery.SchemaField(
                "frames",
                "RECORD",
                mode="REPEATED",
                fields=[
                    bigquery.SchemaField(
                        "timestamp", "STRING", mode="REQUIRED"
                    ),
                    bigquery.SchemaField(
                        "objects",
                        "RECORD",
                        mode="REPEATED",
                        fields=[
                            bigquery.SchemaField(
                                "label", "STRING", mode="REQUIRED"
                            ),
                            bigquery.SchemaField(
                                "box_2d",
                                "RECORD",
                                mode="REQUIRED",
                                fields=[
                                    bigquery.SchemaField(
                                        "xmin", "INTEGER", mode="REQUIRED"
                                    ),
                                    bigquery.SchemaField(
                                        "ymin", "INTEGER", mode="REQUIRED"
                                    ),
                                    bigquery.SchemaField(
                                        "xmax", "INTEGER", mode="REQUIRED"
                                    ),
                                    bigquery.SchemaField(
                                        "ymax", "INTEGER", mode="REQUIRED"
                                    ),
                                ],
                            ),
                            bigquery.SchemaField(
                                "width", "INTEGER", mode="NULLABLE"
                            ),
                            bigquery.SchemaField(
                                "height", "INTEGER", mode="NULLABLE"
                            ),
                            bigquery.SchemaField(
                                "area", "INTEGER", mode="NULLABLE"
                            ),
                            bigquery.SchemaField(
                                "center_x", "FLOAT", mode="NULLABLE"
                            ),
                            bigquery.SchemaField(
                                "center_y", "FLOAT", mode="NULLABLE"
                            ),
                        ],
                    ),
                    bigquery.SchemaField(
                        "object_count", "INTEGER", mode="NULLABLE"
                    ),
                    bigquery.SchemaField(
                        "human_count", "INTEGER", mode="NULLABLE"
                    ),
                    bigquery.SchemaField(
                        "car_count", "INTEGER", mode="NULLABLE"
                    ),
                ],
            ),
            bigquery.SchemaField(
                "sequences",
                "RECORD",
                mode="REPEATED",
                fields=[
                    bigquery.SchemaField(
                        "start_time", "STRING", mode="REQUIRED"
                    ),
                    bigquery.SchemaField(
                        "end_time", "STRING", mode="REQUIRED"
                    ),
                    bigquery.SchemaField(
                        "description", "STRING", mode="NULLABLE"
                    ),
                ],
            ),
        ]

    def ensure_table_exists(self) -> None:
        """Create the BigQuery table if it doesn't exist."""
        try:
            self.client.get_table(self.table_id)
            print(f"Table {self.table_id} already exists")
        except NotFound:
            table = bigquery.Table(self.table_id, schema=self.schema)
            table = self.client.create_table(table, exists_ok=True)
            print(f"Table {self.table_id} created")

    def load_json_data(self, file_path: str) -> dict[str, dict | list]:
        """Load JSON data from file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def process_detection_data(
        self, detection_data: dict[str, dict | list]
    ) -> list[dict[str, dict]]:
        """Process detection data for BigQuery format."""
        transformed_data = []

        for video_path, frames in detection_data.items():
            processed_frames = []

            for frame in frames:
                processed_objects = []

                for obj in frame["objects"]:
                    x1, y1, x2, y2 = obj["box_2d"]
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    processed_obj = {
                        "label": obj["label"],
                        "box_2d": {
                            "xmin": x1,
                            "ymin": y1,
                            "xmax": x2,
                            "ymax": y2,
                        },
                        "width": width,
                        "height": height,
                        "area": area,
                        "center_x": center_x,
                        "center_y": center_y,
                    }
                    processed_objects.append(processed_obj)

                human_count = sum(
                    1 for obj in processed_objects if obj["label"] == "human"
                )
                car_count = sum(
                    1 for obj in processed_objects if obj["label"] == "car"
                )

                processed_frame = {
                    "timestamp": frame["timestamp"],
                    "objects": processed_objects,
                    "object_count": len(processed_objects),
                    "human_count": human_count,
                    "car_count": car_count,
                }
                processed_frames.append(processed_frame)

            record = {
                "video_path": video_path,
                "frames": processed_frames,
                "sequences": [],
            }
            transformed_data.append(record)

        return transformed_data

    def export_detections(self, detection_file: str) -> None:
        """Export detection data to BigQuery."""
        print(f"Loading detection data from {detection_file}")
        detection_data = self.load_json_data(detection_file)

        print("Processing detection data...")
        transformed_data = self.process_detection_data(detection_data)

        print("Ensuring table exists...")
        self.ensure_table_exists()

        jsonl_file = "detections_export.jsonl"
        print(f"Writing data to {jsonl_file}")
        with open(jsonl_file, "w") as f:
            for record in transformed_data:
                f.write(json.dumps(record) + "\n")

        print(f"Loading data to BigQuery table {self.table_id}")
        job_config = bigquery.LoadJobConfig(
            schema=self.schema,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )

        with open(jsonl_file, "rb") as source_file:
            job = self.client.load_table_from_file(
                source_file, self.table_id, job_config=job_config
            )

        job.result()
        print(f"✓ Detection data exported successfully to {self.table_id}")
        Path(jsonl_file).unlink()

    def export_sequences(self, sequences_file: str) -> None:
        """Export sequences data by updating existing table."""
        print(f"Loading sequences data from {sequences_file}")
        sequences_data = self.load_json_data(sequences_file)

        query = f"SELECT * FROM `{self.table_id}`"
        existing_data = list(self.client.query(query))
        existing_records = {row.video_path: dict(row) for row in existing_data}

        updated_records = []
        for video_path, sequences in sequences_data.items():
            if video_path in existing_records:
                record = existing_records[video_path].copy()
                record['sequences'] = sequences
                updated_records.append(record)
                print(f"Prepared sequences for {video_path}")
            else:
                print(f"Warning: {video_path} not found in existing data")

        if not updated_records:
            print("No records to update")
            return

        print(f"Loading {len(updated_records)} updated records to BigQuery...")

        job_config = bigquery.LoadJobConfig(
            schema=self.schema,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        )

        sequences_jsonl = "sequences_update.jsonl"
        with open(sequences_jsonl, "w") as f:
            for record in updated_records:
                f.write(json.dumps(record) + "\n")

        with open(sequences_jsonl, "rb") as source_file:
            job = self.client.load_table_from_file(
                source_file, self.table_id, job_config=job_config
            )

        job.result()

        Path(sequences_jsonl).unlink()
        print(f"All sequences exported successfully to {self.table_id}")
        print(f"Updated {len(updated_records)} records with sequence data")


def main():
    parser = argparse.ArgumentParser(
        description="Export video intelligence data to BigQuery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export detections (creates/replaces table data)
  python export/bq.py --mode detections --input results.json

  # Export sequences (updates existing records)
  python export/bq.py --mode sequences --input sequences.json
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["detections", "sequences"],
        required=True,
        help="Export mode: 'detections' or 'sequences'",
    )

    parser.add_argument("--input", required=True, help="Input JSON file path")

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the config YAML file.",
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)

    try:
        config = Config.from_yaml(args.config)
        exporter = BigQueryExporter(config)

        if args.mode == "detections":
            exporter.export_detections(args.input)
        elif args.mode == "sequences":
            exporter.export_sequences(args.input)

        print("Export completed successfully!")

    except Exception as e:
        print(f"Error during export: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
