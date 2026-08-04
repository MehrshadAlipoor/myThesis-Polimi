"""Global configuration for the evaluation framework.

Holds the OpenAI-compatible client (pointing at a local ``llama-cpp-server``
Docker container), the model registry loaded from ``models.ini``, and the
helper used to restart the container to flush VRAM when swapping judge models.

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
# LLM client
# ---------------------------------------------------------------------
LLAMA_SERVER_URL = os.environ.get(
    "NSCLC_EVAL_SERVER_URL", "http://localhost:8080")
client: OpenAI = OpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="sk-local")

SEED = 42
TEMPERATURE = 0.2

# ---------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------
_registry = configparser.ConfigParser()
_registry.read(MODELS_INI)
available_models = [m for m in _registry.sections() if "bge" not in m.lower()]

JUDGE_MODELS = available_models
JUDGE_MODEL = available_models[0] if available_models else "qwen2.5:7b"


def configure(
    client_instance=None,
    judge_model: str | None = None,
    judge_models: list[str] | None = None,
    seed: int | None = None,
    temperature: float | None = None,
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
    """
    global client, JUDGE_MODEL, JUDGE_MODELS, SEED, TEMPERATURE
    if client_instance is not None:
        client = client_instance
    if judge_model is not None:
        JUDGE_MODEL = judge_model
    if judge_models is not None:
        JUDGE_MODELS = judge_models
    if seed is not None:
        SEED = seed
    if temperature is not None:
        TEMPERATURE = temperature


# ---------------------------------------------------------------------
# Docker VRAM management (llama-cpp-server)
# ---------------------------------------------------------------------
DOCKER_CONTAINER_NAME = "llama-cpp-server"
HEALTH_TIMEOUT = int(os.environ.get("NSCLC_EVAL_HEALTH_TIMEOUT", "120"))
DOCKER_MODEL_VOLUME = os.environ.get(
    "NSCLC_EVAL_DOCKER_VOLUME",
    r"C:\Users\miskovicvanja\Desktop\gabriele\thesis\docker\llama-cpp:/models",
)


def restart_llama_server(model_name: str) -> None:
    """Stop and restart the ``llama-cpp-server`` Docker container.

    Restarting flushes GPU VRAM so a new judge model can be loaded. The model
    is mounted from ``models.ini`` presets and loads lazily on first request.
    """
    print(
        f"🔄 Restarting {DOCKER_CONTAINER_NAME} to clear VRAM for '{model_name}'...")

    # Step 1: Stop and remove old container
    subprocess.run(["docker", "stop", DOCKER_CONTAINER_NAME],
                   capture_output=True)
    subprocess.run(["docker", "rm", DOCKER_CONTAINER_NAME],
                   capture_output=True)
    time.sleep(5)  # Wait for VRAM to be freed

    # Step 2: Start new container with the lab workstation's exact arguments
    print("  Starting container with lab configuration...")
    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", DOCKER_CONTAINER_NAME,
            "--gpus", "all",
            "-v", DOCKER_MODEL_VOLUME,
            "-p", "8080:8080",
            "ghcr.io/ggml-org/llama.cpp:server-cuda",
            "--models-preset", "/models/models.ini",
            "--parallel", "1",
            "--models-max", "2",
            "--flash-attn", "on",
            "--context-shift",
            "--embeddings",
            "--batch-size", "2048",
            "--ubatch-size", "2048",
        ],
        check=True,
    )

    # Step 3: Wait for health check
    for i in range(HEALTH_TIMEOUT):
        try:
            with urllib.request.urlopen(f"{LLAMA_SERVER_URL}/health", timeout=2) as resp:
                if resp.status == 200:
                    print(f"  ✅ llama-cpp-server restarted ({i + 1}s). "
                          f"Model '{model_name}' will load on first inference request.")
                    return
        except Exception:
            pass
        time.sleep(1)

    raise RuntimeError(
        f"llama-cpp-server failed to start within {HEALTH_TIMEOUT}s")
