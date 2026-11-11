# Experimental Apple Silicon MLX backend

MLX supports local LFM3-1.7B text models on Apple Silicon. GGUF models still use llama.cpp.

Use mlx-vlm 0.7.12 or later. Compared with 0.7.11, it fixes output limits, response parsing, tool streams, and long prompt handling. See the [upstream changes](https://github.com/Blaizzy/mlx-vlm/compare/v0.7.11...v0.7.12).

## Setup

Use native arm64 macOS and Node 22.19 or later. Python 3.12 and 3.15 are tested. Install in a virtual environment:

```sh
python -m pip install 'mlx-vlm==0.7.12'
```

Use the [official 4-bit model](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-4bit) from the Hugging Face cache. In DSH Settings → Models → LFM, select its snapshot folder. Set **MLX Python** to the virtual environment's absolute Python path, then choose **Save and load**.

## Behavior

MLX processes prompts in steps of up to 256 tokens. Smaller context caps are available, including 4K, 8K and 16K. The native context remains the default.

Prompt and output tokens share the context cap. Oversized requests fail before streaming. The server keeps DSH history unchanged and uses the profile's output limit unless the request sets one.

Vision, quantized KV cache, memory fitting, and MLX benchmarks are unsupported. Model loading runs offline and does not execute checkpoint Python code.

## Validation

Run from the repository root with the cached snapshot folder and virtual environment's Python interpreter:

```sh
node scripts/validate-mlx.mjs MODEL_DIRECTORY VENV_PYTHON
```

Use the existing cache for reproduction. Do not call `snapshot_download()`.

All 18 unit tests passed. Apple Silicon smoke tests passed for text, Thinking mode, output limits, context overflow, longer prompts, cancellation, tools, and unload.

Full-context performance, memory pressure, GPU cancellation timing, and DSH UI cancellation remain unmeasured. A successful tool round trip does not establish reliable tool selection.
