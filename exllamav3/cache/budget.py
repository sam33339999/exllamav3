"""Size a reserved KV pool from free memory, without allocating it.

The pool is created once at load and kept until the process exits. This
module only does the arithmetic.
"""
from __future__ import annotations

from pathlib import Path

from ..constants import PAGE_SIZE

GIB = 1024 ** 3


def kv_pool_tokens(
    *,
    free_bytes: int,
    weight_bytes: int,
    bytes_per_token: float,
    util: float = 0.85,
    reserve_gb: float = 16.0,
    max_gb: float = 48.0,
    scratch_gb: float = 4.0,
    page_size: int = PAGE_SIZE,
) -> int:
    """Token capacity of the reserved pool.

    leave = max(reserve, free * (1 - util))
    budget = min(free - weights - scratch - leave, max_gb)
    The result is floored to a whole number of pages. A budget below one
    page raises RuntimeError so startup stops instead of loading a pool
    that cannot hold a single page.
    """
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")
    if not 0 < util <= 1:
        raise ValueError(f"KV_UTIL must be in (0, 1], got {util}")
    if free_bytes < 0 or weight_bytes < 0:
        raise ValueError("free_bytes and weight_bytes must be >= 0")
    leave = max(reserve_gb * GIB, free_bytes * (1.0 - util))
    budget = free_bytes - weight_bytes - scratch_gb * GIB - leave
    budget = min(budget, max_gb * GIB)
    page_bytes = page_size * bytes_per_token
    if budget < page_bytes:
        raise RuntimeError(
            f"KV budget is {budget / GIB:.2f} GiB, below one page "
            f"({page_size} tokens, {page_bytes / GIB:.4f} GiB). "
            f"free={free_bytes / GIB:.2f} GiB, weights={weight_bytes / GIB:.2f} GiB, "
            f"leave={leave / GIB:.2f} GiB"
        )
    tokens = int(budget // bytes_per_token)
    tokens -= tokens % page_size
    if tokens < page_size:
        raise RuntimeError(
            f"KV budget rounds down below one page of {page_size} tokens"
        )
    return tokens


def cache_bytes_per_token(cache) -> float:
    """Bytes of paged KV per token. Recurrent slots are not included."""
    if cache.max_num_tokens <= 0:
        raise ValueError("cache max_num_tokens must be positive")
    total = 0
    for layer in cache.layers.values():
        total += int(layer.storage_size())
    if total <= 0:
        raise ValueError("cache reports no KV storage")
    return total / cache.max_num_tokens


def safetensors_bytes(model_dir: str | Path) -> int:
    root = Path(model_dir)
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no safetensors weights in {root}")
    return sum(p.stat().st_size for p in files)


def language_max_position(config) -> int | None:
    """Token limit of the language model.

    Vision wrappers often leave the outer ``max_position_embeddings`` at the
    8192 default while the real limit lives under ``text_config``.
    """
    if config is None:
        return None
    raw = getattr(config, "config_dict", None)
    if isinstance(raw, dict):
        text = raw.get("text_config")
        if isinstance(text, dict) and text.get("max_position_embeddings"):
            return int(text["max_position_embeddings"])
    outer = getattr(config, "max_position_embeddings", None)
    if outer:
        return int(outer)
    return None


def prompt_too_long(num_tokens: int, max_position_embeddings: int | None) -> str | None:
    """Error text when a prompt walks past the model's position table."""
    if not max_position_embeddings or max_position_embeddings <= 0:
        return None
    limit = int(max_position_embeddings)
    if num_tokens > limit:
        return (
            f"Prompt is {num_tokens} tokens, past this model's "
            f"max_position_embeddings ({limit})"
        )
    return None
