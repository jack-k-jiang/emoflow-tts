import torch
from f5_tts.api import F5TTS

from emoflow_tts.inference.benchmark import (
    audio_duration_seconds,
    compute_real_time_factor,
    timed_inference,
)
from emoflow_tts.models.base import (
    InferenceMetadata,
    InferenceWrapper,
    SynthesisRequest,
    SynthesisResult
)

class F5Wrapper(InferenceWrapper):
    def __init__(self, model_name: str = "F5TTS_v1_Base", device: str = "cuda") -> None:
        self._device = device
        self._model_name = model_name
        self._model = F5TTS(model=model_name, device=device)

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        with timed_inference(device=self._device) as timing:
            wav, sample_rate, _spec = self._model.infer(
                ref_file=request.reference_audio_path,
                ref_text=request.reference_transcript,
                gen_text=request.text,
                nfe_step=request.sampling_steps,
                seed=request.seed,
            )

        audio = torch.from_numpy(wav)
        duration = audio_duration_seconds(audio.shape[-1], sample_rate)
        rtf = compute_real_time_factor(timing["elapsed_seconds"], duration)

        metadata = InferenceMetadata(
            model_name=self._model_name,
            model_version = None,
            device = self._device,
            precision=request.precision,
            sampling_steps=request.sampling_steps,
            reference_duration_seconds=duration,
            emotion_condition=request.emotion,
            seed=request.seed,
            inference_time_seconds=timing["elapsed_seconds"],
            generated_audio_duration_seconds=duration,
            real_time_factor=rtf,
        )
        return SynthesisResult(audio=audio, sample_rate=sample_rate, metadata=metadata)