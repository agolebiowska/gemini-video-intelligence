from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class PathsConfig:
    preprocessing_config: str
    tmp: str
    metadata: str
    ground_truth: str
    results: str
    sequences: str
    detection_eval_results: str
    sequence_eval_results: str


@dataclass
class Config:
    project_id: str
    location: str
    bucket: str
    model_id: str
    dataset_id: str
    table_id: str
    input_path: str
    preprocessed_path: str
    output_path: str
    objects_to_detect: list[str]
    detection_prompt: str
    max_items_to_detect: int
    debug_interval: int
    sequences_prompt: str
    paths: PathsConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        project_root = Path.cwd().parent
        resolved_paths = {
            key: project_root / value for key, value in cfg['paths'].items()
        }

        paths_config = PathsConfig(**resolved_paths)

        return cls(
            project_id=cfg['project_id'],
            location=cfg['location'],
            bucket=cfg['bucket'],
            model_id=cfg['model_id'],
            dataset_id=cfg['dataset_id'],
            table_id=cfg['table_id'],
            input_path=cfg['input_path'],
            preprocessed_path=cfg['preprocessed_path'],
            output_path=cfg['output_path'],
            objects_to_detect=cfg['objects_to_detect'],
            detection_prompt=cfg['detection_prompt'],
            max_items_to_detect=cfg['max_items_to_detect'],
            debug_interval=cfg['debug_interval'],
            sequences_prompt=cfg['sequences_prompt'],
            paths=paths_config,
        )
