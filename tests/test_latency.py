import pytest
from src.emoflow_tts.benchmark import compute_real_time_factor

def test_computer_RTF():
    assert compute_real_time_factor(2.0, 4.0) == 0.5

def test_divide_by_zero_exception():
    with pytest.raises(ValueError):
        compute_real_time_factor(2.0, 0.0)