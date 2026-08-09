from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch

def compute_real_time_factor(
        inference_time_seconds: float,
        audio_duration_seconds: float,
) -> float:
    """RTF = wall-clock inference time / generated-audio duration.
    
    RTF < 1.0 means synthesis is faster than real-time.
    """
    if audio_duration_seconds <= 0:
        raise ValueError(f"audio_duration_seconds must be > 0, got {audio_duration_seconds}")
    if inference_time_seconds < 0:
        raise ValueError(f"inference_time_seconds must be >= 0, got {inference_time_seconds}")
    return inference_time_seconds / audio_duration_seconds

def audio_duration_seconds(num_samples: int, sample_rate: int) -> float:
    """Duration, in seconds, of a generated audio tensor."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
    return num_samples / sample_rate

@contextmanager
def timed_inference(device: str | torch.device = "cpu"):
    """Times a block of code, synchronizing CUDA so async kernels are
       actually finished before the clock stops.

    Usage: 
        timing = {}
        with timed_inference(device="cuda") as timing:
            audio = model.generate(...)
        elapsed = timing["elapsed_seconds"]
    """
    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    result: dict[str, float] = {}
    if is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield result
    finally:
        if is_cuda:
            torch.cuda.synchronize()
        result["elapsed_seconds"] = time.perf_counter() - start

def test_computer_RTF():
    assert compute_real_time_factor(2.0, 4.0) == 0.5