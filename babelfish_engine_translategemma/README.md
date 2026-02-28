# babelfish_engine_translategemma

TranslateGemma engine for BabelFish Redo (Luanti mod)

## Description

This module provides a translation engine based on TranslateGemma - an open machine translation model from Google, built on Gemma 3. TranslateGemma supports translation between 55+ languages.

## Installation

1. Install the module in your Luanti server's `mods/` folder
2. Add `babelfish_engine_translategemma` to `secure.http_mods` in `minetest.conf`:

```conf
secure.http_mods = babelfish_engine_translategemma
```

## Configuration

Add the following parameters to `minetest.conf`:

### Basic Settings

```conf
# API type: ollama (default), vllm, hf, openai
babelfish_engine_translategemma.api_type = ollama

# API URL (default depends on API type)
babelfish_engine_translategemma.serviceurl = http://localhost:11434/api/generate

# Model name (default: translategemma)
babelfish_engine_translategemma.model = translategemma

# Enable verbose logging (default: false)
babelfish_engine_translategemma.debug = true

# Enable full model info dump including layer tensors (default: false, very large output)
# Only use this if you need to see all model tensor details
babelfish_engine_translategemma.verbose_debug = false
```

### Generation Parameters (Ollama)

```conf
# Temperature (0.0 - 2.0, default: 0.3)
babelfish_engine_translategemma.temperature = 0.3

# Maximum tokens to generate (default: 1024)
babelfish_engine_translategemma.num_predict = 1024

# Context size (default: 2048)
babelfish_engine_translategemma.num_ctx = 2048

# Top-K sampling (default: 40)
babelfish_engine_translategemma.top_k = 40

# Top-P sampling (default: 0.9)
babelfish_engine_translategemma.top_p = 0.9

# Number of layers on GPU (-1 = auto, default: -1)
babelfish_engine_translategemma.num_gpu = -1
```

### Hugging Face Token (only for "hf" API type)

```conf
babelfish_engine_translategemma.hf_token = hf_xxxxx
```

## Log Information

At startup, the module outputs detailed information about Ollama:

```
[babelfish_engine_translategemma] ========== Ollama Startup Info ==========
[babelfish_engine_translategemma] Ollama API URL: http://localhost:11434
[babelfish_engine_translategemma] Model name: translategemma
[babelfish_engine_translategemma] Ollama version: 0.1.27
[babelfish_engine_translategemma] Model: translategemma
[babelfish_engine_translategemma] Model format: gguf
[babelfish_engine_translategemma] Model family: gemma
[babelfish_engine_translategemma] Parameter count: 4B
[babelfish_engine_translategemma] Quantization: Q4_K_M
[babelfish_engine_translategemma] Architecture: gemma
[babelfish_engine_translategemma] Parameters: 4.2B
[babelfish_engine_translategemma] Context length: 8192
[babelfish_engine_translategemma] GPU acceleration: ENABLED (35 layers on GPU)
[babelfish_engine_translategemma] No models currently loaded in memory
[babelfish_engine_translategemma] =======================================
```

### For Each Translation Request:

```
[babelfish_engine_translategemma] Translation request: en -> ru, text length: 45 chars
[babelfish_engine_translategemma] Response in 1.23s (load: 0.15s, eval: 1.08s), tokens: prompt=52, generated=38, speed: 35.2 tok/s
[babelfish_engine_translategemma] Translation successful: 156 chars output
```

### Response Parameters:

| Field | Description |
|-------|-------------|
| `total_duration` | Total request time |
| `load_duration` | Model loading time |
| `eval_duration` | Generation time |
| `prompt_eval_count` | Number of tokens in prompt |
| `eval_count` | Number of generated tokens |
| `speed` | Generation speed (tokens/sec) |

## Supported APIs

### Ollama (Recommended)

```conf
babelfish_engine_translategemma.api_type = ollama
babelfish_engine_translategemma.serviceurl = http://localhost:11434/api/generate
```

Requirements:
1. Install [Ollama](https://ollama.ai)
2. Download the TranslateGemma model:
   ```bash
   ollama pull translategemma
   ```

### vLLM

```conf
babelfish_engine_translategemma.api_type = vllm
babelfish_engine_translategemma.serviceurl = http://localhost:8000/v1/completions
```

### Hugging Face Inference API

```conf
babelfish_engine_translategemma.api_type = hf
babelfish_engine_translategemma.serviceurl = https://api-inference.huggingface.co/models/google/translategemma-4b
babelfish_engine_translategemma.hf_token = hf_xxxxx
```

### OpenAI-Compatible API

```conf
babelfish_engine_translategemma.api_type = openai
babelfish_engine_translategemma.serviceurl = http://localhost:8000/v1/chat/completions
```

## Supported Languages

TranslateGemma supports the following languages (55+):

ar, bg, bn, ca, cs, da, de, el, en, es, et, fa, fi, fr, he, hi, hr, hu, id, it, ja, ko, lt, lv, ms, nl, no, pl, pt, ro, ru, sk, sl, sr, sv, th, tl, tr, uk, vi, zh, af, am, az, be, bs, cy, eo, eu, gl, gu, hy, ka, kn, ml, mr, ne, pa, si, sq, sw, ta, te, ur, uz

## Language Aliases

- `zh`, `zhs`, `zh-cn`, `zh-hans`, `chinese` → Chinese
- `en`, `english` → English
- `ru`, `russian` → Russian
- `de`, `german` → German
- `fr`, `french` → French
- `es`, `spanish` → Spanish
- `ja`, `japanese` → Japanese
- `ko`, `korean` → Korean

## Checking GPU Acceleration

### Method 1: Via Minetest Logs

When starting the server with `debug = true`, you will see:
```
[babelfish_engine_translategemma] GPU acceleration: ENABLED (35 layers on GPU)
```

Or if GPU is disabled:
```
[babelfish_engine_translategemma] GPU acceleration: DISABLED (0 layers on GPU)
```

### Method 2: Via Ollama API

```bash
# Check version
curl http://localhost:11434/api/version

# Model information
curl -X POST http://localhost:11434/api/show -d '{"name":"translategemma"}'

# Loaded models
curl http://localhost:11434/api/ps
```

### Method 3: Force GPU Enable

```conf
# -1 = auto (default)
# 0 = CPU only
# 1+ = specific number of layers on GPU (e.g., 99 for all)
babelfish_engine_translategemma.num_gpu = 99
```

## Comparison with babelfish_engine_lingva

| Feature | Lingva | TranslateGemma |
|---------|--------|----------------|
| API Type | GraphQL | REST (LLM-style) |
| Requires Internet | Yes | No (local) |
| External Service Dependency | Yes | No |
| Speed | Fast | Depends on hardware |
| Privacy | Data on server | Full privacy |
| Translation Quality | Google Translate | Gemma 3 based |
| GPU Acceleration | N/A | Yes |

## Requirements

- Minetest 5.4+ or Luanti
- babelfish_redo (main mod)
- Running inference server (Ollama, vLLM, or HF API access)

## Troubleshooting

### "Could not fetch model info"
Model is not installed. Run:
```bash
ollama pull translategemma
```

### "GPU acceleration: DISABLED"
Check:
1. GPU drivers installed (NVIDIA CUDA / AMD ROCm)
2. GPU available to Ollama: `nvidia-smi`
3. Environment variable: `CUDA_VISIBLE_DEVICES`

### Slow Generation
1. Ensure GPU is enabled
2. Increase context if needed: `num_ctx = 4096`
3. Lower temperature for faster responses

## License

LGPL-3.0-or-later

## Authors

- Original babelfish_engine_lingva: Tai "DuCake" Kedzierski, 1F616EMO
- TranslateGemma adaptation: 2026 Eugeny aka Bad
