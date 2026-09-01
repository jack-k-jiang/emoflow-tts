from importlib.resources import files
import json
from dataclasses import asdict
from pathlib import Path

import torchaudio

from emoflow_tts.config import load_config
from emoflow_tts.models.base import SynthesisRequest
from emoflow_tts.models.f5_wrapper import F5Wrapper

repo_root = Path(__file__).parent.parent
config = load_config(repo_root / "configs/inference/baseline.yaml")

wrapper = F5Wrapper(model_name=config.model.name, device=config.model.device)

# Hardcoded for this single ad-hoc test — in the real batch pipeline these
# come from a ManifestRow (data.manifest_path), not the experiment config.
ref = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
request = SynthesisRequest(
    text="This is a test of the emotion-aware text to speech system.",
    reference_audio_path=ref,
    reference_transcript="Some call me nature, others call me mother nature.",
    emotion=config.inference.emotion,
    emotion_strength=config.inference.emotion_strength,
    sampling_steps=config.inference.sampling_steps,
    seed=config.experiment.seed,
    precision=config.model.precision,
)
result = wrapper.synthesize(request)
print(result.metadata.real_time_factor, result.metadata.inference_time_seconds)

output_dir = repo_root / config.data.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

audio_path = output_dir / "sample_001.wav"
# .unsqueeze(0) to change (samples,)  ->  (channesl, samples)
torchaudio.save(str(audio_path), result.audio.unsqueeze(0).float(), result.sample_rate)

metadata_path = output_dir / "sample_001.json"
metadata_path.write_text(json.dumps(asdict(result.metadata), indent=2))

print(f"Saved audio to {audio_path}")
print(f"Saved metadata to {metadata_path}")