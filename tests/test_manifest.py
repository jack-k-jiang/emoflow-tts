import pytest
from pydantic import ValidationError

from emoflow_tts.data.manifest import ManifestRow


def test_valid_manifest_row():
    row = ManifestRow(
        utterance_id = "utt_001", 
        audio_path = "data/processed/utt_001.wav",
        transcript = "hello world",
        speaker_id = "spk_01",
        emotion = "neutral",
        split="train",
        sample_rate=24000,
        duration_seconds = 2.5,
        dataset_source="dev_subset",
    )
    assert row.utterance_id == "utt_001"

def test_invalid_split_rejected():
    with pytest.raises(ValidationError):
        ManifestRow(
            utterance_id="utt_002",
            audio_path="x.wav",
            transcript="hi",
            speaker_id="spk_01",
            emotion="neutral",
            split="training",  # not one of train/val/test
            sample_rate=24000,
            duration_seconds=1.0,
            dataset_source="dev_subset",
        )

def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        ManifestRow(audio_path="x.wav")