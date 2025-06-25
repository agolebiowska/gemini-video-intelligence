import argparse
import json
import sys
from google.cloud import bigquery

from video_intelligence.config import Config
from video_intelligence.utils import io
from video_intelligence.processing.prompts import PROMPTS
from video_intelligence.utils.llm import LLMCaller


class SequenceProcessor:
    """
    Analyzes a video using its object detection data to extract key moments.
    """

    def __init__(self, config: Config):
        self._config = config
        self._llm_caller = LLMCaller(
            model=config.model_id, project_id=config.project_id
        )
        self._bq_client = bigquery.Client(project=config.project_id)

    def _get_detections_from_bq(self, video_gcs_path: str) -> list[dict]:
        sql = f"""
        SELECT
            frame.timestamp,
            ARRAY_AGG(DISTINCT obj.label) as objects
        FROM
            `{self._config.project_id}.{self._config.dataset_id}.{self._config.table_id}`,
            UNNEST(frames) as frame, UNNEST(frame.objects) as obj
        WHERE
            video_path = '{video_gcs_path}'
        GROUP BY
            frame.timestamp ORDER BY frame.timestamp
        """
        query_job = self._bq_client.query(sql)
        return [
            {"timestamp": row.timestamp, "objects": list(row.objects)}
            for row in query_job
        ]

    def extract_sequences(
        self, video_bytes: bytes, detections: list[dict]
    ) -> list[dict] | None:
        """
        Generates key sequences for a video using Gemini.

        Args:
            video_bytes: The video content as bytes.
            detections: A list of object detections for the video.

        Returns:
            A list of dictionaries, where each dictionary represents a key sequence.
        """
        if not detections:
            print("Warning: No detections provided. Cannot extract sequences.")
            return None

        prompt_config = PROMPTS[self._config.sequences_prompt]

        formatted_prompt = prompt_config["prompt"].format(
            detections=json.dumps(detections, indent=2)
        )

        try:
            response = self._llm_caller.call_llm(
                data=video_bytes,
                prompt=formatted_prompt,
                response_schema=prompt_config["response_schema"],
                mime_type="video/mp4",
            )
            return response.content
        except Exception as e:
            print(f"Error calling LLM for sequence extraction: {e}")
            return None

    @staticmethod
    def save_results(results: dict[str, list[dict]], output_file: str) -> None:
        with open(output_file, "w") as f:
            f.write(json.dumps(results, indent=2))
        print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate video sequence descriptions for videos.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the config YAML file.",
    )
    parser.add_argument(
        "--output",
        default="sequences.json",
        help="Output file to save results (JSON format). Default: sequences.json",
    )

    args = parser.parse_args()

    try:
        config = Config.from_yaml(args.config)
        gcs_io = io.GcsIO(config.project_id, f"gs://{config.bucket}")
        videos = gcs_io.list_files(path=config.paths.input)[1:]
        videos = [
            f"{config.paths.preprocessed}/{video.split('/')[-1].replace('.mov', '')}/{video.split('/')[-1].replace('.mov', '.mp4')}"
            for video in videos
        ]
        processor = SequenceProcessor(config)

        if not videos:
            print("No videos found in the specified input path. Exiting.")
            sys.exit(0)

        all_results = {}
        for gcs_path in videos:
            try:
                detections = processor._get_detections_from_bq(gcs_path)
                if not detections:
                    print(
                        f"Warning: No detections found for {gcs_path}. Skipping."
                    )
                    all_results[gcs_path] = {}
                    continue

                video_bytes = gcs_io.get_video(path=gcs_path)
                sequences = processor.extract_sequences(
                    video_bytes, detections
                )
                all_results[gcs_path] = sequences
                print(f"Successfully extracted sequences for {gcs_path}.")

            except Exception as e:
                print(f"Failed to process {gcs_path}: {e}", file=sys.stderr)
                all_results[gcs_path] = {"error": str(e)}

        if args.output:
            SequenceProcessor.save_results(all_results, args.output)

        print("\nSequence extraction completed successfully!")

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
