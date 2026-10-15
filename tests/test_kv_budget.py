"""KV pool sizing and the per-sequence position limit. No GPU allocation."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import pytest

from exllamav3.cache.budget import (
    GIB,
    cache_bytes_per_token,
    kv_pool_tokens,
    prompt_too_long,
)
from exllamav3.cache.nvfp4 import CacheLayer_nvfp4
from exllamav3.constants import PAGE_SIZE


def test_pool_leaves_reserve_and_aligns_to_page():
    free = 80 * GIB
    weights = int(15.5 * GIB)
    # 5760 bytes/token, one page is an exact multiple.
    tokens = kv_pool_tokens(
        free_bytes = free,
        weight_bytes = weights,
        bytes_per_token = 5760,
        util = 0.85,
        reserve_gb = 16,
        max_gb = 48,
        scratch_gb = 4,
    )
    # leave = max(16, 80 * 0.15) = 16 GiB
    # budget = 80 - 15.5 - 4 - 16 = 44.5 GiB, under the 48 GiB cap
    budget = 44.5 * GIB
    expect = int(budget // 5760)
    expect -= expect % PAGE_SIZE
    assert tokens == expect
    assert tokens % PAGE_SIZE == 0
    assert tokens * 5760 <= budget


def test_pool_hard_cap():
    tokens = kv_pool_tokens(
        free_bytes = 200 * GIB,
        weight_bytes = 15 * GIB,
        bytes_per_token = 5760,
        util = 0.85,
        reserve_gb = 16,
        max_gb = 48,
        scratch_gb = 4,
    )
    assert tokens * 5760 <= 48 * GIB
    assert tokens > 40 * GIB / 5760


def test_pool_refuses_a_budget_below_one_page():
    with pytest.raises(RuntimeError, match = "below one page"):
        kv_pool_tokens(
            free_bytes = 20 * GIB,
            weight_bytes = 15 * GIB,
            bytes_per_token = 5760,
            util = 0.85,
            reserve_gb = 16,
            max_gb = 48,
            scratch_gb = 4,
        )


def test_pool_rejects_bad_util():
    with pytest.raises(ValueError, match = "KV_UTIL"):
        kv_pool_tokens(
            free_bytes = 80 * GIB,
            weight_bytes = GIB,
            bytes_per_token = 5760,
            util = 0,
        )


def test_bytes_per_token_uses_layer_storage():
    class _Layer:
        def storage_size(self):
            return 5760 * 256

    cache = SimpleNamespace(max_num_tokens = 256, layers = {"a": _Layer()})
    assert cache_bytes_per_token(cache) == 5760


def test_nvfp4_minicpm_layer_is_576_bytes_per_token():
    attn = SimpleNamespace(num_kv_heads = 2, head_dim = 256)
    layer = CacheLayer_nvfp4(None, attn, 0, PAGE_SIZE)
    assert layer.storage_size() / PAGE_SIZE == 576


def test_prompt_limit():
    assert prompt_too_long(100, 262144) is None
    assert prompt_too_long(262144, 262144) is None
    message = prompt_too_long(262145, 262144)
    assert message is not None and "262145" in message
    assert prompt_too_long(10 ** 9, None) is None
    assert prompt_too_long(10 ** 9, 0) is None
