import torch

from emoflow_tts.inference.benchmark import audio_duration_seconds, compute_real_time_factor
from emoflow_tts.models.base import (
    InferenceMetadata,
    InferenceWrapper,
    SynthesisRequest,
    SynthesisResult,
)


class FakeWrapper(InferenceWrapper):
    """A stand-in InferenceWrapper for testing, with no real model behind it.

    F5Wrapper needs a GPU and a multi-hundred-MB checkpoint download to run
    at all, which makes it unusable in CI or for a quick local test. Because
    InferenceWrapper is an abstract interface (see base.py), anything that
    calls .synthesize() doesn't care *which* implementation it's talking to —
    so this class can stand in for F5Wrapper anywhere the *plumbing* (request
    in, correctly-shaped result out) is what's being tested, not the model
    itself.

    It returns silent audio instead of real speech, but still builds a real,
    correctly-typed SynthesisResult — including real timing and a real RTF
    calculation — so tests can check that the surrounding code (metadata
    assembly, RTF math, result handling) is actually correct.
    """

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        # Fixed for simplicity — a real wrapper would get this from the model.
        sample_rate = 24000

        # 2 seconds of silence stands in for "generated audio". Shape matches
        # what F5Wrapper produces: 1D, (num_samples,).
        audio = torch.zeros(sample_rate * 2)

        # Pretend inference took half a second — real wrapper measures this
        # with timed_inference() around the actual model call.
        inference_time_seconds = 0.5

        duration = audio_duration_seconds(audio.shape[-1], sample_rate)
        rtf = compute_real_time_factor(inference_time_seconds, duration)

        metadata = InferenceMetadata(
            model_name="fake",
            model_version=None,
            device="cpu",
            precision=request.precision,
            sampling_steps=request.sampling_steps,
            reference_duration_seconds=duration,
            emotion_condition=request.emotion,
            seed=request.seed,
            inference_time_seconds=inference_time_seconds,
            generated_audio_duration_seconds=duration,
            real_time_factor=rtf,
        )

        return SynthesisResult(audio=audio, sample_rate=sample_rate, metadata=metadata)
