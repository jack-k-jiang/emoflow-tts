import pytest
from pydantic import ValidationError

from emoflow_tts.config import load_config

VALID_YAML = """
experiment:
    name: baseline_f5
    seed: 42
model:
    name: f5_tts
    device: cuda
    precision: fp16
inference:
    sampling_steps: 16
    emotion: neutral
    emotion_strength: 1.0
    reference_duration_seconds: 5.0
data:
    manifest_path: data/manifests/test.csv
    output_dir: outputs/audio
evaluation:
    compute_wer: true
    compute_speaker_similarity: true
    compute_emotion_accuracy: true
    compute_latency: true
"""

def test_load_valid_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(VALID_YAML)
    config = load_config(config_path)
    assert config.experiment.seed == 42
    assert config.inference.sampling_steps == 16

def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")

def test_invalid_config_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("experiment:\n  name: broken\n")  # missing required fields
    with pytest.raises(ValidationError):
        load_config(config_path)