from importlib.resources import files

from emoflow_tts.models.f5_wrapper import F5Wrapper
from emoflow_tts.models.base import SynthesisRequest

wrapper = F5Wrapper(device="cuda")

ref = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
request = SynthesisRequest(
    text="This is a test of the emotion-aware text to speech system.",
    reference_audio_path=ref,
    reference_transcript="Some call me nature, others call me mother nature.",
    emotion=None,
    emotion_strength=1.0,
    sampling_steps=16,
    seed=42,
    precision="fp16",
)
result = wrapper.synthesize(request)
print(result.metadata.real_time_factor, result.metadata.inference_time_seconds)
