from pathlib import Path

import yaml
from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    name: str
    seed: int

class ModelConfig(BaseModel):
    name: str
    checkpoint: str | None = None
    device: str
    precision: str

class InferenceConfig(BaseModel):
    sampling_steps: int
    emotion: str
    emotion_strength: float
    reference_duration_seconds: float

class DataConfig(BaseModel):
    manifest_path: str
    output_dir: str

class EvaluationConfig(BaseModel):
    compute_wer: bool
    compute_speaker_similarity: bool
    compute_emotion_accuracy: bool
    compute_latency: bool

class Config(BaseModel):
    experiment: ExperimentConfig
    model: ModelConfig
    inference: InferenceConfig
    data: DataConfig
    evaluation: EvaluationConfig

def load_config(path: str | Path) -> Config:
    """Load and validate a YAML experiment configuration file.

    Args:
        path: Path to a YAML file with `experiment`, `model`, `inference`,
            `data`, and `evaluation` top-level sections.

    Returns:
        A validated Config instance.

    Raises:
        FileNotFoundError: If path does not point to an existing file.
        pydantic.ValidationError: If the YAML content doesn't match the
            expected schema.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)