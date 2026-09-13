"""LLM call infrastructure.

A global semaphore bounds in-flight requests to the server's ``--parallel``
slot count; transient errors are retried with backoff; truncated generations
get one automatic ``max_tokens`` doubling (up to a ceiling); and thinking
judges get a per-request ``thinking_budget_tokens`` cap that bounds their
hidden chain-of-thought without turning it off.

Per-patient log lines are kept contiguous even when several patients are
evaluated concurrently: workers write into a thread-local buffer which the
parent flushes atomically.
"""

from __future__ import annotations

import builtins
import io
import threading
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    LengthFinishReasonError,
    RateLimitError,
)

from . import config

_LLM_SEM = threading.BoundedSemaphore(max(1, config.PARALLEL_SLOTS))
_LLM_RETRIES = 4
_MAX_TOKENS_CEILING = 8192
_LLM_TIMING_LOG = False
_TRANSIENT = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)


def _with_slot(fn, **kwargs):
    last_exc = None
    for attempt in range(_LLM_RETRIES):
        try:
            # Cap the hidden CoT for thinking judges (no-op otherwise).
            budget = config._thinking_budget_for(kwargs.get("model"))
            if budget is not None:
                extra = dict(kwargs.get("extra_body") or {})
                extra["thinking_budget_tokens"] = budget
                kwargs["extra_body"] = extra

            t0 = time.time()
            with _LLM_SEM:
                resp = fn(**kwargs)
            if _LLM_TIMING_LOG:
                usage = getattr(resp, "usage", None)
                completion = getattr(usage, "completion_tokens", None) if usage else None
                print(f"  ⏱ {kwargs.get('model')} {time.time() - t0:6.1f}s "
                      f"max_tokens={kwargs.get('max_tokens')} completion_tokens={completion} "
                      f"thinking_budget={budget}")
            return resp
        except LengthFinishReasonError:
            new_cap = min(int(kwargs.get("max_tokens") or 1024) * 2, _MAX_TOKENS_CEILING)
            print(f"  ⚠️ Output truncated at max_tokens={kwargs.get('max_tokens')}; "
                  f"retrying with max_tokens={new_cap}")
            kwargs["max_tokens"] = new_cap
            last_exc = None
        except _TRANSIENT as exc:
            last_exc = exc
            wait = 10 * (attempt + 1)
            print(f"  ⚠️ LLM call failed ({type(exc).__name__}), "
                  f"retry {attempt + 1}/{_LLM_RETRIES} in {wait}s")
            time.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM call kept hitting the token cap even after retries")


def parse_call(**kwargs):
    """Structured-output call under the global semaphore, with retries."""
    return _with_slot(config.client.beta.chat.completions.parse, **kwargs)


def create_call(**kwargs):
    """Plain chat call under the global semaphore, with retries."""
    return _with_slot(config.client.chat.completions.create, **kwargs)


# ---------------------------------------------------------------------
# Per-patient log routing
# ---------------------------------------------------------------------
_orig_print = builtins.print
_print_lock = threading.Lock()
_tls = threading.local()


def _print_router(*args, **kwargs):
    buf = getattr(_tls, "buffer", None)
    if buf is not None and "file" not in kwargs:
        _orig_print(*args, **kwargs, file=buf)
    else:
        with _print_lock:
            _orig_print(*args, **kwargs)


builtins.print = _print_router


def new_patient_buffer() -> io.StringIO:
    """Create a fresh per-patient log buffer."""
    return io.StringIO()


def bind_print_buffer(buf, fn):
    """Route a function's prints into a parent patient buffer even when it runs
    on an inner worker thread (keeps per-patient log blocks contiguous)."""
    def wrapper(*args, **kwargs):
        prev = getattr(_tls, "buffer", None)
        _tls.buffer = buf
        try:
            return fn(*args, **kwargs)
        finally:
            _tls.buffer = prev
    return wrapper


def flush_patient_buffer(buf) -> None:
    """Atomically write a finished patient's buffered log to stdout."""
    if buf is None:
        return
    with _print_lock:
        _orig_print(buf.getvalue())
