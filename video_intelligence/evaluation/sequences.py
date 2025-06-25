import json
import argparse
import sys

import pandas as pd
import vertexai
from vertexai.evaluation import (
    EvalTask,
    EvalResult,
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
)
from google.cloud import bigquery

from video_intelligence.config import Config


class SequenceEvaluator:
    """Evaluates video sequence descriptions using Vertex AI Evaluation Service."""

    def __init__(self, config: Config):
        self.config = config
        vertexai.init(
            project=self.config.project_id, location=self.config.location
        )
        self.bq_client = bigquery.Client(project=self.config.project_id)
        self.table_id = f"{self.config.project_id}.{self.config.dataset_id}.{self.config.table_id}"

    @property
    def evaluation_metric(self) -> PointwiseMetric:
        """Define the evaluation metric for sequence description quality."""
        return PointwiseMetric(
            metric="video_description_quality",
            metric_prompt_template=PointwiseMetricPromptTemplate(
                criteria={
                    "object_detection_accuracy": (
                        "Descriptions clearly prioritize and accurately reflect the detected objects (humans, vehicles, infrastructure) presented in the data. Each sequence focuses primarily on what was actually detected rather than inferring or assuming objects not in the detection data."
                    ),
                    "sequence_coherence": (
                        "Sequences are divided at meaningful transition points based on significant changes in object presence or activity. Each sequence has clear start/end timestamps with descriptions that logically connect the objects' appearance, movements, and interactions."
                    ),
                },
                rating_rubric={
                    "2": (
                        "The response excels at both criteria: accurately describing detected objects as the primary focus and creating coherent, well-defined sequences based on meaningful object transitions."
                    ),
                    "1": (
                        "The response is adequate on both criteria but has minor issues: either occasionally focusing on undetected elements or creating sequences with some unclear transitions/timestamps."
                    ),
                    "0": (
                        "The response has significant issues: either frequently describing undetected objects/activities or creating poorly defined sequences that don't align with meaningful object transitions."
                    ),
                    "-1": (
                        "The response fails on both criteria: descriptions don't prioritize detected objects and sequences are arbitrary or confusing with inaccurate timestamps."
                    ),
                },
            ),
        )

    def _prepare_evaluation_dataset(self) -> pd.DataFrame:
        """Prepare the evaluation dataset."""
        query = f"""
        WITH VideoData AS (
          SELECT
            t.video_path,
            frame
          FROM
            `{self.table_id}` AS t,
            UNNEST(frames) AS frame
        ),
        AggregatedObjects AS (
          SELECT
            video_path,
            ARRAY_AGG(DISTINCT obj.label ORDER BY obj.label) AS all_objects
          FROM
            VideoData,
            UNNEST(frame.objects) AS obj
          GROUP BY
            video_path
        ),
        AggregatedSequences AS (
          SELECT
            t.video_path,
            ARRAY_AGG(
              STRUCT(seq.start_time, seq.end_time, seq.description)
              ORDER BY seq.start_time
            ) AS sequences
          FROM
            `{self.table_id}` AS t,
            UNNEST(sequences) AS seq
          GROUP BY
            t.video_path
        )
        SELECT
          ao.video_path,
          ao.all_objects,
          aseq.sequences
        FROM AggregatedObjects AS ao
        LEFT JOIN AggregatedSequences AS aseq ON ao.video_path = aseq.video_path
        """

        try:
            results_df = self.bq_client.query(query).to_dataframe()

            if results_df.empty:
                return pd.DataFrame()

            def format_response(row):
                objects_data = []
                sequences_data = []

                if row["all_objects"] is not None:
                    objects_data = list(row["all_objects"])

                if row["sequences"] is not None:
                    sequences_data = [
                        {
                            "start_time": s["start_time"],
                            "end_time": s["end_time"],
                            "description": s["description"],
                        }
                        for s in row["sequences"]
                    ]

                result_dict = {
                    "objects": objects_data,
                    "sequences": sequences_data,
                }
                return json.dumps(result_dict)

            results_df["response"] = results_df.apply(format_response, axis=1)

            eval_dataset = results_df[["video_path", "response"]].rename(
                columns={"video_path": "video_id"}
            )

            print(
                f"Created evaluation dataset with {len(eval_dataset)} video analyses"
            )
            return eval_dataset

        except Exception as e:
            print(f"Error preparing evaluation dataset: {e}")
            return pd.DataFrame()

    def run_evaluation(
        self,
        experiment_name: str = "scene-understanding-evaluation",
    ) -> EvalResult:
        """Run the evaluation as a Vertex AI Experiment."""
        print(f"Running experiment: {experiment_name}")

        try:
            eval_dataset = self._prepare_evaluation_dataset()

            eval_result = EvalTask(
                dataset=eval_dataset,
                metrics=[self.evaluation_metric],
                experiment=experiment_name,
            ).evaluate()

            print("✓ Evaluation completed successfully")
            return eval_result

        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise

    @staticmethod
    def display_results(eval_result: EvalResult) -> None:
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(eval_result)

    @staticmethod
    def save_results(eval_result: EvalResult, output_file: str) -> None:
        try:
            results_data = {
                "summary_metrics": getattr(eval_result, "summary_metrics", {}),
                "metrics_table": (
                    getattr(
                        eval_result, "metrics_table", pd.DataFrame()
                    ).to_dict("records")
                    if hasattr(eval_result, "metrics_table")
                    else []
                ),
                "experiment_name": getattr(
                    eval_result, "experiment", "unknown"
                ),
            }

            with open(output_file, "w") as f:
                json.dump(results_data, f, indent=2, default=str)

            print(f"Results saved to {output_file}")

        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate video sequence descriptions using Vertex AI Evaluation Service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the config YAML file.",
    )

    parser.add_argument(
        "--experiment",
        default="scene-understanding-pointwise-evaluation",
        help="Experiment name for the evaluation",
    )

    args = parser.parse_args()

    try:
        config = Config.from_yaml(args.config)
        evaluator = SequenceEvaluator(config)

        eval_result = evaluator.run_evaluation(args.experiment)
        evaluator.display_results(eval_result)
        SequenceEvaluator.save_results(
            eval_result, config.paths.sequence_evaluation_results
        )

        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
