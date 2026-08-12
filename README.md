# Emoflow-TTS
> Emoflow is a emotion-aware, low-latency Text-to-Speech.

Setup
-----

Requires Python 3.10 or 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# Install torch/torchaudio matching your GPU's CUDA version first —
# the default PyPI wheel may not match (e.g. on a Lightning.ai T4/L40S/A100 Studio):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install the package and its remaining dependencies:
pip install -r requirements.txt
```

References
-------------

### Packaging & tooling

- [Python Packaging User Guide — pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [Hatchling build backend docs](https://hatch.pypa.io/latest/config/build/)
- [pytest docs — Getting started](https://docs.pytest.org/en/stable/getting-started.html)

### Configuration & data modeling

- [Pydantic v2 docs — Models](https://docs.pydantic.dev/latest/concepts/models/)
- [Pydantic v2 docs — Validators](https://docs.pydantic.dev/latest/concepts/validators/)
- [PyYAML docs](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [pandas docs — I/O tools (read_csv / read_json)](https://pandas.pydata.org/docs/reference/io.html)
- [jsonlines package docs](https://jsonlines.readthedocs.io/)
- [Python docs — abc module](https://docs.python.org/3/library/abc.html)
- [Python docs — typing.Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [Python docs — dataclasses](https://docs.python.org/3/library/dataclasses.html)

### F5-TTS

- [SWivid/F5-TTS — official repository](https://github.com/swivid/f5-tts)
- [f5-tts on PyPI](https://pypi.org/project/f5-tts/)
- [Running F5-TTS Locally for Voice Cloning — setup guide](https://builderai.tools/blog/running-f5-tts-locally-for-voice-cloning)