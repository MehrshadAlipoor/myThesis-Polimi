"""Global configuration for the evaluation framework.

Holds the OpenAI-compatible client (pointing at a local ``llama-cpp-server``
Docker container), the model registry loaded from ``models.ini``, and the
helper used to restart the container to flush VRAM when swapping judge models.

The speed-edition additions live here too:

* ``MAX_TOKENS`` — per-metric output caps.
* ``THINKING_BUDGET_TOKENS`` — per-request cap on hidden chain-of-thought for
  thinking judges (CoT stays ON, only its length is bounded).
* ``LLAMA_CPP_IMAGE`` / ``LLAMA_SERVER_FLAGS`` — pinned server image and flags.

Paths and the model registry can be overridden through environment variables,
and a mock client can be injected for offline testing / demo notebooks.
"""

from __future__ import annotations

import configparser
import os
import subprocess
import time
import urllib.request

from openai import OpenAI

# ---------------------------------------------------------------------
# Paths (overridable via environment variables)
# ---------------------------------------------------------------------
DATA_DIR = os.environ.get("NSCLC_EVAL_DATA_DIR", "data/sample")
GROUND_TRUTH_CSV = os.environ.get(
    "NSCLC_EVAL_GT_CSV", "data/sample/sample_ground_truth.csv")
MODELS_INI = os.environ.get("NSCLC_EVAL_MODELS_INI", "config/models.ini")

# ---------------------------------------------------------------------
# LLM client (lazy)
# ---------------------------------------------------------------------
# Constructing ``OpenAI()`` eagerly builds an httpx transport whose SSL context
# walks the OS trust store — slow (and offline-hostile) at import time. The
# real client is therefore created on first use; ``configure`` can inject a
# mock for tests and demos.
LLAMA_SERVER_URL = os.environ.get(
    "NSCLC_EVAL_SERVER_URL", "http://localhost:8080")
CLIENT_TIMEOUT = float(os.environ.get("NSCLC_EVAL_TIMEOUT", "900"))

_client = None


def _build_client() -> OpenAI:
    return OpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="sk-local",
                  timeout=CLIENT_TIMEOUT)


def _get_client():
    global _client
    if _client is None:
        _client = _build_client()
    return _client


class _LazyClient:
    """Forwards attribute access to the lazily-built OpenAI client."""

    def __getattr__(self, item):
        return getattr(_get_client(), item)


client = _LazyClient()

SEED = 42
TEMPERATURE = 0.2

# Concurrency: must match the llama-cpp-server "--parallel" slot count.
PARALLEL_SLOTS = int(os.environ.get("NSCLC_EVAL_PARALLEL", "2"))

# ---------------------------------------------------------------------
# Thinking budget (capped CoT)
# ---------------------------------------------------------------------
# Caps the hidden reasoning tokens the thinking judges emit BEFORE the JSON
# answer. Thinking stays ON — only its length is bounded.
#   * TWEAK: change a value (e.g. 1024 -> 2048) to give a judge more room.
#   * TOGGLE OFF per judge: set its value to None or -1.
#   * TOGGLE OFF globally: set THINKING_BUDGET_ENABLED = False.
THINKING_BUDGET_ENABLED = True
THINKING_BUDGET_TOKENS = {
    "baichuan-m2-32b": 1024,
    "qwen3-30B-A3B-Thinking": 1024,
}


def _thinking_budget_for(model_name) -> int | None:
    """Thinking-token cap for a judge, or None to leave generation untouched."""
    if not THINKING_BUDGET_ENABLED:
        return None
    val = THINKING_BUDGET_TOKENS.get(model_name)
    if val is None or val < 0:
        return None
    return int(val)


# ---------------------------------------------------------------------
# Output-token caps per metric
# ---------------------------------------------------------------------
MAX_TOKENS = {
    "decompose": 2048,
    "classify": 4096,
    "completeness": 2048,
    "factuality": 4096,
    "faithfulness": 2048,
    "gar": 4096,
    "binary": 512,
    "treat_comp": 2048,
    "out_fact": 4096,
}

# Context truncation caps (characters) for text fed to the judges.
GUIDELINE_CHAR_CAP = 8000
PATIENT_DATA_CHAR_CAP = 6000

# ---------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------
_registry = configparser.ConfigParser()
_registry.read(MODELS_INI)
available_models = [
    m for m in _registry.sections()
    if "bge" not in m.lower() and "medgemma" not in m.lower()
]

JUDGE_MODELS = available_models
JUDGE_MODEL = available_models[0] if available_models else "qwen2.5:7b"


def configure(
    client_instance=None,
    judge_model: str | None = None,
    judge_models: list[str] | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    parallel_slots: int | None = None,
    thinking_budget_enabled: bool | None = None,
) -> None:
    """Override global settings, e.g. to inject a mock client for offline demos.

    Parameters
    ----------
    client_instance :
        Any object exposing the OpenAI ``chat.completions`` interface. When
        provided it replaces the live client (useful for tests / demos).
    judge_model :
        The default judge model name used for single-model calls.
    judge_models :
        The full list of judge models used by ``run_all_patient_evaluations``.
    seed, temperature :
        Sampling parameters forwarded to every LLM call.
    parallel_slots :
        Override the concurrency bound (defaults to ``NSCLC_EVAL_PARALLEL``).
    thinking_budget_enabled :
        Toggle the capped-CoT behaviour globally.
    """
    global _client, JUDGE_MODEL, JUDGE_MODELS, SEED, TEMPERATURE
    global PARALLEL_SLOTS, THINKING_BUDGET_ENABLED
    if client_instance is not None:
        _client = client_instance
    if judge_model is not None:
        JUDGE_MODEL = judge_model
    if judge_models is not None:
        JUDGE_MODELS = judge_models
    if seed is not None:
        SEED = seed
    if temperature is not None:
        TEMPERATURE = temperature
    if parallel_slots is not None:
        PARALLEL_SLOTS = int(parallel_slots)
    if thinking_budget_enabled is not None:
        THINKING_BUDGET_ENABLED = bool(thinking_budget_enabled)


# ---------------------------------------------------------------------
# Docker VRAM management (llama-cpp-server)
# ---------------------------------------------------------------------
DOCKER_CONTAINER_NAME = "llama-cpp-server"
HEALTH_TIMEOUT = int(os.environ.get("NSCLC_EVAL_HEALTH_TIMEOUT", "900"))

# Pinned server image (override with NSCLC_EVAL_IMAGE). The build must be recent
# enough to support the per-request ``thinking_budget_tokens`` field.
LLAMA_CPP_IMAGE = os.environ.get(
    "NSCLC_EVAL_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda")

# Flags applied at container start. No server-level --reasoning-budget: that
# would disable the per-request thinking_budget_tokens injection.
LLAMA_SERVER_FLAGS = {
    "n_gpu_layers": 999,
    "ctx_size": 32768,
    "flash_attn": "on",
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "batch_size": 2048,
    "ubatch_size": 512,
    "cache_prompt": True,
}

# Snapshot recorded in every evaluation JSON for methodology traceability.
SERVER_CONFIG_SNAPSHOT = {
    "image": LLAMA_CPP_IMAGE,
    "parallel_slots": PARALLEL_SLOTS,
    "flags": dict(LLAMA_SERVER_FLAGS),
    "thinking_budget_enabled": THINKING_BUDGET_ENABLED,
    "thinking_budget_tokens": dict(THINKING_BUDGET_TOKENS),
    "max_tokens": dict(MAX_TOKENS),
}

DOCKER_MODEL_VOLUME = os.environ.get(
    "NSCLC_EVAL_DOCKER_VOLUME",
    r"C:\Users\miskovicvanja\Desktop\gabriele\thesis\docker\llama-cpp:/models",
)


def _model_gguf_path(model_name: str) -> str:
    """Resolve the GGUF path for a judge from ``models.ini``."""
    try:
        return _registry.get(model_name, "model")
    except (configparser.NoSectionError, configparser.NoOptionError):
        raise RuntimeError(
            f"models.ini has no usable [section] for '{model_name}'. "
            f"Available sections: {_registry.sections()}")


def restart_llama_server(model_name: str) -> None:
    """Stop and restart the ``llama-cpp-server`` Docker container.

    A full restart flushes GPU VRAM so a new judge model can be loaded. The
    model is launched explicitly with ``--model/--alias`` so every server flag
    is guaranteed to apply (models-preset mode silently reverts ctx-size).
    """
    gguf_path = _model_gguf_path(model_name)
    print(f"🔄 Restarting {DOCKER_CONTAINER_NAME} to flush VRAM for '{model_name}'...")

    subprocess.run(["docker", "stop", DOCKER_CONTAINER_NAME], capture_output=True)
    subprocess.run(["docker", "rm", DOCKER_CONTAINER_NAME], capture_output=True)
    time.sleep(5)

    print(f"  Starting llama-cpp-server (single model, parallel={PARALLEL_SLOTS}, "
          f"ctx={LLAMA_SERVER_FLAGS['ctx_size']}, image={LLAMA_CPP_IMAGE})...")
    run_args = [
        "docker", "run", "-d",
        "--name", DOCKER_CONTAINER_NAME,
        "--gpus", "all",
        "-p", "8080:8080",
        "-v", DOCKER_MODEL_VOLUME,
        LLAMA_CPP_IMAGE,
        "--model", gguf_path,
        "--alias", model_name,
        "--parallel", str(PARALLEL_SLOTS),
        "--ctx-size", str(LLAMA_SERVER_FLAGS["ctx_size"]),
        "--flash-attn", LLAMA_SERVER_FLAGS["flash_attn"],
        "--cache-type-k", LLAMA_SERVER_FLAGS["cache_type_k"],
        "--cache-type-v", LLAMA_SERVER_FLAGS["cache_type_v"],
        "--n-gpu-layers", str(LLAMA_SERVER_FLAGS["n_gpu_layers"]),
        "--embeddings",
        "--batch-size", str(LLAMA_SERVER_FLAGS["batch_size"]),
        "--ubatch-size", str(LLAMA_SERVER_FLAGS["ubatch_size"]),
    ]
    if LLAMA_SERVER_FLAGS.get("cache_prompt"):
        run_args.append("--cache-prompt")
    subprocess.run(run_args, check=True)

    for i in range(HEALTH_TIMEOUT):
        try:
            with urllib.request.urlopen(f"{LLAMA_SERVER_URL}/health", timeout=2) as resp:
                if resp.status == 200:
                    print(f"  ✅ llama-cpp-server healthy ({i + 1}s). "
                          f"Model '{model_name}' loads on first inference request.")
                    return
        except Exception:
            pass
        time.sleep(1)

    raise RuntimeError(
        f"llama-cpp-server failed to start within {HEALTH_TIMEOUT}s")
