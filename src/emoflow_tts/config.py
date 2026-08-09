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