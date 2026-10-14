---
title: Use a Local Model - Zed
description: Configure llama.cpp, Ollama, LM Studio, local OpenAI-compatible servers, and local edit prediction in Zed.
---

# Use a Local Model

Use local models when you run the model on your machine or on infrastructure you control.

| Local path                        | Zed AI features      | External Agents | Terminal Threads | Notes                                              |
| --------------------------------- | -------------------- | --------------- | ---------------- | -------------------------------------------------- |
| llama.cpp                         | Yes                  | Separate config | Separate config  | Configure a llama.cpp server for Zed AI features   |
| LM Studio                         | Yes                  | Separate config | Separate config  | Configure LM Studio for Zed AI features            |
| Ollama                            | Yes                  | Separate config | Separate config  | Configure Ollama for Zed AI features               |
| Local OpenAI-compatible server    | Yes                  | Separate config | Separate config  | Configure base URL, model, and key if needed       |
| Local/self-hosted edit prediction | Edit Prediction only | No              | No               | Uses [Edit Prediction](./edit-prediction.md) setup |

## llama.cpp {#llama-cpp}

Use [llama.cpp](https://llama.app) and its built-in server for local models with Zed Agent, Inline Assistant, and similar model-backed Zed AI features.

1. Install llama.cpp from [llama.app](https://llama.app).
2. Start the server in [router mode](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md):

   ```sh
   llama serve
   ```

   It loads models from the llama.cpp cache on demand. To download and run a specific model in one step, pass `-hf`:

   ```sh
   llama serve -hf unsloth/gemma-4-26B-A4B-it-GGUF:BF16
   ```

3. In Zed, select a llama.cpp model from the model dropdown.

Zed automatically discovers the served models with their context length and tool/vision capabilities. In router mode these are refined once a model loads, via the server's `/models/sse` stream (which requires a recent llama.cpp build). To list models yourself instead, set `auto_discover` to `false`:

```json [settings]
{
  "language_models": {
    "llama.cpp": {
      "api_url": "http://localhost:8080",
      "auto_discover": false,
      "available_models": [
        {
          "name": "gemma-4-12b-it-GGUF:BF16",
          "display_name": "gemma-4-12b-it-GGUF:BF16",
          "max_tokens": 32768,
          "supports_tools": true,
          "supports_images": false
        }
      ]
    }
  }
}
```

### llama.cpp Context Length {#llama-cpp-context}

Zed uses the context length the server reports (`/props`). Override it for all models with `context_window`, or per model with `max_tokens` in `available_models`:

```json [settings]
{
  "language_models": {
    "llama.cpp": {
      "context_window": 8192
    }
  }
}
```

If your llama.cpp server requires a key, enter it in the provider UI or set `LLAMACPP_API_KEY`. For a remote server, set the API URL to its endpoint and provide the key (set on the server with `--api-key`).

## Ollama {#ollama}

Use Ollama for local models with Zed Agent, Inline Assistant, and similar model-backed Zed AI features.

1. Download and install Ollama from [ollama.com/download](https://ollama.com/download).
2. Pull a model:

   ```sh
   ollama pull mistral
   ```

3. Make sure the Ollama server is running. On macOS, open Ollama.app. On Linux or from a shell, run:

   ```sh
   ollama serve
   ```

4. In Zed, select an Ollama model from the model dropdown.

Zed automatically discovers models that Ollama has pulled. To disable autodiscovery and list models yourself, configure `auto_discover`:

```json [settings]
{
  "language_models": {
    "ollama": {
      "api_url": "http://localhost:11434",
      "auto_discover": false,
      "available_models": [
        {
          "name": "qwen2.5-coder",
          "display_name": "qwen 2.5 coder",
          "max_tokens": 32768,
          "supports_tools": true,
          "supports_thinking": true,
          "supports_images": true
        }
      ]
    }
  }
}
```

### Ollama Context Length {#ollama-context}

By default, Zed omits `num_ctx` from Ollama requests so that Ollama manages
the context size using its server configuration and model parameters. Zed does
not request the model's theoretical maximum context length automatically.

To explicitly request a context length for all Ollama models:

```json [settings]
{
  "language_models": {
    "ollama": {
      "context_window": 8192
    }
  }
}
```

You can also configure context length per model with `max_tokens` in
`available_models`. The global `context_window` takes precedence. Both must be
positive integers. To restore server management, remove both overrides; model
entries may omit `max_tokens` when only customizing names or capabilities.

Zed uses the explicit context setting as its conversation budget when present.
Otherwise, it uses a recent matching runtime capacity from Ollama's `/api/ps`,
then its last observed capacity for the same model identity, the model's default
`num_ctx`, or a conservative estimate of 4096 tokens.
Budgets are bounded by the model's advertised context limit. Runtime observations
are refreshed every 60 seconds without loading models. If the model unloads or
the query fails, Zed keeps the last observed budget, labelled runtime unconfirmed,
rather than resetting an existing conversation to 4096. A newly reported capacity
replaces that budget; changing provider settings clears observations.
The estimate is not sent to Ollama and is not a guarantee against input truncation
or out-of-memory errors. This does not add automatic compaction for small-context
models or dynamically resize memory as tokens arrive.

### Ollama Model Residency {#ollama-residency}

Zed also omits `keep_alive` by default, leaving idle unloading to Ollama rather
than keeping models loaded indefinitely. Set `keep_alive` in a model's
`available_models` entry to override this: `0` unloads after the request, `-1`
keeps the model loaded indefinitely, and a duration such as `"5m"` sets an idle
timeout. Explicit existing settings continue to apply. Ollama's server settings
may still keep a model loaded indefinitely even when Zed omits the parameter.

If your Ollama server requires a key, enter the key in the provider UI or set `OLLAMA_API_KEY`. For remote Ollama services such as Ollama Turbo, set the API URL to the remote endpoint and provide an API key.

## LM Studio {#lm-studio}

Use LM Studio for local models with Zed Agent, Inline Assistant, and similar model-backed Zed AI features.

1. Download and install [LM Studio](https://lmstudio.ai/download).
2. Download at least one model in LM Studio, or use the LM Studio CLI:

   ```sh
   lms get qwen2.5-coder-7b
   ```

3. Start the LM Studio API server:

   ```sh
   lms server start
   ```

4. In Zed, select an LM Studio model from the model dropdown.

If your LM Studio server requires a key, enter the key in the provider UI or set `LMSTUDIO_API_KEY`.

## Local OpenAI-Compatible Servers {#openai-compatible}

Use [OpenAI-compatible endpoints](./use-api-access.md#openai-compatible) for local or self-hosted servers that expose an OpenAI-compatible API.

## Local Edit Prediction {#edit-prediction}

Edit Prediction has its own provider setup. See [Edit Prediction](./edit-prediction.md) for local and self-hosted edit prediction options.

## Agent Path Boundaries {#agent-path-boundaries}

This page covers local models configured in Zed. [External Agents](./external-agents.md) and terminal CLIs may have their own local-model setup; configure those in the agent or CLI.
