#!/usr/bin/env python3
"""Unit tests for tools/serve_openai.py thinking-budget helpers.

No GPU: these cover request parsing, think-cap math, and </think> splitting
so a long think cannot swallow max_tokens and return an empty answer.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "serve_openai", ROOT / "tools" / "serve_openai.py")
so = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(so)


class TestSplitReasoning(unittest.TestCase):
    def test_split_closed_think(self):
        r, c = so.split_reasoning("aaa</think>\n\nhello")
        self.assertEqual(r, "aaa")
        self.assertEqual(c, "hello")

    def test_split_unclosed_think_is_all_reasoning(self):
        r, c = so.split_reasoning("still thinking about it")
        self.assertEqual(r, "")
        self.assertEqual(c, "still thinking about it")

    def test_split_unclosed_with_open_tag(self):
        r, c = so.split_reasoning("<think>\nstill thinking")
        self.assertEqual(r, "still thinking")
        self.assertEqual(c, "")

    def test_inject_then_split_yields_content(self):
        text = "long think" + so.THINK_CLOSE_INJECT + "the answer"
        r, c = so.split_reasoning(text)
        self.assertEqual(r, "long think")
        self.assertEqual(c, "the answer")


class TestThinkLimits(unittest.TestCase):
    def test_xhigh_caps_at_effort_budget(self):
        cap, reserve = so.resolve_think_limits(
            16384, True, "xhigh", None, 512)
        self.assertEqual(reserve, 512)
        self.assertEqual(cap, 8192)

    def test_client_1024_still_reserves_answer(self):
        # Typical OpenAI-client default. Old server spent all 1024 on think.
        cap, reserve = so.resolve_think_limits(
            1024, True, "xhigh", None, 512)
        self.assertEqual(reserve, 512)
        self.assertEqual(cap, 512)
        self.assertGreater(1024 - cap, 0)

    def test_tiny_max_tokens_still_leaves_room(self):
        cap, reserve = so.resolve_think_limits(
            200, True, "xhigh", None, 512)
        self.assertEqual(reserve, 50)
        self.assertEqual(cap, 150)
        self.assertEqual(cap + reserve, 200)

    def test_explicit_budget_smaller_than_effort(self):
        cap, reserve = so.resolve_think_limits(
            16384, True, "xhigh", 64, 512)
        self.assertEqual(cap, 64)
        self.assertEqual(reserve, 512)

    def test_budget_zero_means_only_reserve(self):
        cap, reserve = so.resolve_think_limits(
            4096, True, "xhigh", 0, 512)
        self.assertEqual(reserve, 512)
        self.assertEqual(cap, 4096 - 512)

    def test_thinking_off_has_no_cap(self):
        cap, reserve = so.resolve_think_limits(
            1024, False, "xhigh", None, 512)
        self.assertIsNone(cap)
        self.assertEqual(reserve, 0)

    def test_force_close_at_cap(self):
        self.assertFalse(so.should_force_close_think(True, 63, 64))
        self.assertTrue(so.should_force_close_think(True, 64, 64))
        self.assertFalse(so.should_force_close_think(False, 999, 64))
        self.assertFalse(so.should_force_close_think(True, 999, None))


class TestParseRequest(unittest.TestCase):
    def _msgs(self, **extra):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        body.update(extra)
        return body

    def test_default_max_tokens_is_no_longer_1024(self):
        req, err = so.parse_request(self._msgs())
        self.assertIsNone(err)
        self.assertEqual(req["max_tokens"], so.DEFAULT_MAX_TOKENS)
        self.assertGreaterEqual(req["max_tokens"], 8192)
        self.assertTrue(req["enable_thinking"])
        self.assertEqual(req["reasoning_effort"], "xhigh")

    def test_enable_thinking_false_string(self):
        req, err = so.parse_request(self._msgs(enable_thinking = "false"))
        self.assertIsNone(err)
        self.assertFalse(req["enable_thinking"])

    def test_chat_template_kwargs(self):
        req, err = so.parse_request(self._msgs(
            chat_template_kwargs = {
                "enable_thinking": False,
                "reasoning_effort": "low",
            }))
        self.assertIsNone(err)
        self.assertFalse(req["enable_thinking"])
        self.assertEqual(req["reasoning_effort"], "low")

    def test_openai_reasoning_effort_object(self):
        req, err = so.parse_request(self._msgs(reasoning = {"effort": "medium"}))
        self.assertIsNone(err)
        self.assertEqual(req["reasoning_effort"], "medium")

    def test_invalid_effort_is_400(self):
        req, err = so.parse_request(self._msgs(reasoning_effort = "ultra"))
        self.assertIsNone(req)
        self.assertIn("reasoning_effort", err)

    def test_thinking_budget_from_body(self):
        req, err = so.parse_request(self._msgs(thinking_budget = 128))
        self.assertIsNone(err)
        self.assertEqual(req["thinking_budget"], 128)

    def test_max_completion_tokens_alias(self):
        req, err = so.parse_request(self._msgs(max_completion_tokens = 333))
        self.assertIsNone(err)
        self.assertEqual(req["max_tokens"], 333)


class TestAsBool(unittest.TestCase):
    def test_common_falsey(self):
        for v in (False, "false", "False", "0", "off", "no"):
            self.assertFalse(so._as_bool(v, True), msg = repr(v))

    def test_common_truthy(self):
        for v in (True, "true", "1", "yes", "on"):
            self.assertTrue(so._as_bool(v, False), msg = repr(v))


if __name__ == "__main__":
    unittest.main()
