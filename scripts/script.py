from importlib.resources import files

from emoflow_tts.config import load_config
from emoflow_tts.models.base import SynthesisRequest
from emoflow_tts.models.f5_wrapper import F5Wrapper

config = load_config("configs/inference/baseline.yaml")

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
