import argparse
import json
import os
import pandas as pd
import ast

from video_intelligence.config import Config
from video_intelligence.preprocessing.preprocessor import (
    VideoPreprocessor,
    VideoConfig,
)
from video_intelligence.processing.detections import VideoProcessor
from video_intelligence.export.bq import BigQueryExporter
from video_intelligence.processing.sequences import SequenceProcessor
from video_intelligence.evaluation.detections import ObjectDetectionEvaluator
from video_intelligence.evaluation.sequences import SequenceEvaluator
from video_intelligence.utils import io, ocv
from video_intelligence.processing.types import Crop


class Pipeline:
    def __init__(self, config_path: str):
        self.config = Config.from_yaml(config_path)
        self.gcs_io = io.GcsIO(
            project_id=self.config.project_id,
            root_path=f"gs://{self.config.bucket}",
        )
        self.local_io = io.LocalIO(root_path=self.config.paths.tmp)
        if not os.path.exists(self.config.paths.tmp):
            os.makedirs(self.config.paths.tmp)

        self.bq_exporter = BigQueryExporter(self.config)
        self.preprocessed_videos_paths = []

    def _preprocess(self):
        print("Starting preprocessing...")
        preproc_df = pd.read_csv(self.config.paths.preprocessing_config)

        for _, row in preproc_df.iterrows():
            video_path = row['video_path']
            start_frame = int(row['start_frame'])

            video_content = self.gcs_io.get_video(video_path)
            temp_input_path = os.path.join(
                self.config.paths.tmp, "temp_preproc_input.mp4"
            )
            self.local_io.save_video(video_content, "temp_preproc_input.mp4")
            cap = ocv.init_video_capture(temp_input_path)
            cap.release()
            os.remove(temp_input_path)

            crop_params = ast.literal_eval(row['crop_params'])
            crop_width, crop_height, top, bottom, left, right = (
                crop_params[0],
                crop_params[1],
                crop_params[2],
                crop_params[3],
                crop_params[4],
                crop_params[5],
            )

            crop = Crop(
                width=crop_width,
                height=crop_height,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
            )

            prep_config = VideoConfig(crop=crop, target_fps=1, max_frames=10)
            preprocessor = VideoPreprocessor(
                source_io=self.gcs_io,
                target_io=self.gcs_io,
                config=prep_config,
            )
            preprocessor.preprocess(
                source_path=video_path,
                target_path=self.config.paths.preprocessed,
                start_frame=start_frame,
            )
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.preprocessed_videos_paths.append(
                f"{self.config.paths.preprocessed}/{video_name}/{video_name}.mp4"
            )
        print("Preprocessing finished.")

    def _process(self):
        print("Starting processing...")
        processor = VideoProcessor(config=self.config)

        results = {}
        for gcs_path in self.preprocessed_videos_paths:
            video_content = self.gcs_io.get_video(path=gcs_path)
            video_name = os.path.basename(gcs_path)
            local_path = os.path.join(self.config.paths.tmp, video_name)
            self.local_io.save_video(video_content, video_name)

            output_path, timestamps = processor.process_video(
                video_path=local_path
            )
            results[gcs_path] = timestamps

            if output_path:
                with open(output_path, "rb") as f:
                    content = f.read()

                processed_gcs_path = gcs_path.replace(
                    self.config.preprocessed_path, self.config.output_path
                )
                self.gcs_io.save_video(content, processed_gcs_path)

        with open(self.config.paths.results, 'w', encoding='utf-8') as f:
            serializable_results = {
                video_path: [ts.to_dict() for ts in timestamps]
                for video_path, timestamps in results.items()
                if timestamps
            }
            json.dump(serializable_results, f, indent=4)
        print("Processing finished.")
        return results

    def _export_detections_to_bq(self):
        print("Exporting detections to BigQuery...")
        self.bq_exporter.export_detections(self.config.paths.results)
        print("Detections exported to BigQuery.")

    def _extract_sequences(self):
        print("Starting scene understanding...")
        sequence_processor = SequenceProcessor(config=self.config)
        sequences = {}
        for gcs_path in self.preprocessed_videos_paths:
            video_bytes = self.gcs_io.get_video(path=gcs_path)
            detections = sequence_processor._get_detections_from_bq(gcs_path)
            response = sequence_processor.extract_sequences(
                video_bytes, detections
            )
            sequences[gcs_path] = response

        with open(self.config.paths.sequences, "w") as f:
            f.write(json.dumps(sequences, indent=2))
        print("Scene understanding finished.")

    def _export_sequences_to_bq(self):
        print("Exporting sequences to BigQuery...")
        self.bq_exporter.export_sequences(self.config.paths.sequences)
        print("Sequences exported to BigQuery.")

    def _evaluate_detections(self):
        print("Starting detections evaluation...")
        detection_evaluator = ObjectDetectionEvaluator(
            ground_truth_file=self.config.paths.ground_truth,
            predictions_file=self.config.paths.results,
        )
        detection_results = detection_evaluator.run_evaluation()
        detection_evaluator.display_results(detection_results)
        detection_evaluator.save_results(
            detection_results, self.config.paths.detection_eval_results
        )
        print("Detections evaluation finished.")

    def _evaluate_sequences(self, video_ids):
        print("Starting sequences evaluation...")
        sequence_evaluator = SequenceEvaluator(self.config)
        eval_result = sequence_evaluator.run_evaluation()
        sequence_evaluator.display_results(eval_result)
        sequence_evaluator.save_results(
            eval_result, self.config.paths.sequence_eval_results
        )
        print("Sequences evaluation finished.")

    def run(self):
        self._preprocess()
        results = self._process()
        self._export_detections_to_bq()
        self._extract_sequences()
        self._export_sequences_to_bq()
        self._evaluate_detections()
        self._evaluate_sequences(list(results.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the video intelligence pipeline."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the config YAML file.",
    )
    args = parser.parse_args()

    pipeline = Pipeline(args.config)
    pipeline.run()
