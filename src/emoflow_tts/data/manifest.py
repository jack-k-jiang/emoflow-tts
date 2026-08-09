from pydantic import AfterValidator, BaseModel
from typing import Literal
import pandas

class ManifestRow(BaseModel):
    # required
    utterance_id: str
    audio_path: str
    transcript: str
    speaker_id: str
    emotion: str
    split: Literal["train", "val", "test"]
    sample_rate: int
    duration_seconds: float
    dataset_source: str

    # optional
    emotion_intensity: float | None = None
    gender: str | None = None
    language: str | None = None
    reference_audio_path: str | None = None
    license: str | None = None
    notes: str | None = None