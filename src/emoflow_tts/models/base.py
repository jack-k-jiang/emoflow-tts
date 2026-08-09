import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class SynthesisRequest:
    text: str
    reference_audio_path: str
    reference_transcript: str | None
    emotion: str | None
    emotion_strength: float
    sampling_steps: int
    seed: int
    precision: str

@dataclass
class InferenceMetadata:
    model_name: str
    model_version: str | None
    device: str
    precision: str
    sampling_steps: int
    reference_duration_seconds: float
    emotion_condition: str | None
    seed: int
    inference_time_seconds: float          # wall-clock
    generated_audio_duration_seconds: float
    real_time_factor: float

@dataclass
class SynthesisResult:
    audio: torch.Tensor
    sample_rate: int
    metadata: InferenceMetadata

class InferenceWrapper(ABC):
    @abstractmethod
    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        ...