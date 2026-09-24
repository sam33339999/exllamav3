"""Think-exit, think-only frequency, and DRY. Imports the steps the server runs."""
import importlib.util
import unittest
from pathlib import Path

import torch

import exllamav3.generator.sampler.custom as sampler_custom
from exllamav3.generator.sampler.custom import (
    SS, SS_DRY, SS_RepP, SS_ThinkExit, SamplingState,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "serve_openai", ROOT / "tools" / "serve_openai.py")
so = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(so)


def _think(budget = 4, ramp = 4, max_bias = 16.0, prompt_len = 5, **kw):
    return SS_ThinkExit(99, prompt_len = prompt_len, budget = budget, ramp = ramp,
                        max_bias = max_bias, **kw)


class TestThinkExit(unittest.TestCase):
    def test_no_bonus_when_close_tag_is_already_in_the_prompt(self):
        past = torch.tensor([[1, 99, 2, 3, 4, 5, 6, 7, 8, 9]])
        self.assertEqual(_think()._bias(past), 0)

    def test_no_bonus_when_close_tag_is_in_the_generated_tokens(self):
        past = torch.tensor([[1, 2, 3, 4, 5, 7, 99, 8, 9, 10, 11]])
        self.assertEqual(_think()._bias(past), 0)

    def test_no_bonus_while_generated_length_is_within_the_budget(self):
        # 5 prompt tokens, then exactly `budget` generated tokens, think still open.
        past = torch.tensor([[1, 2, 3, 4, 5, 7, 8, 9, 10]])
        self.assertEqual(_think()._bias(past), 0)
        # Served budget: 4096 generated tokens is still inside the budget.
        served = _think(budget = 4096, ramp = 1536)
        past = torch.tensor([[1, 2, 3, 4, 5] + [7] * 4096])
        self.assertEqual(served._bias(past), 0)

    def test_bonus_is_half_of_16_halfway_through_the_1536_ramp_and_caps(self):
        served = _think(budget = 4096, ramp = 1536, max_bias = 16)
        halfway = torch.tensor([[1, 2, 3, 4, 5] + [7] * (4096 + 768)])
        self.assertAlmostEqual(served._bias(halfway), 8.0, places = 5)
        past_ramp = torch.tensor([[1, 2, 3, 4, 5] + [7] * (4096 + 1536 + 10)])
        self.assertAlmostEqual(served._bias(past_ramp), 16.0, places = 5)
        # One more generated token past the halfway point raises the bonus.
        longer = torch.tensor([[1, 2, 3, 4, 5] + [7] * (4096 + 769)])
        self.assertGreater(served._bias(longer), served._bias(halfway))

    def test_bonus_rises_after_the_budget_and_caps(self):
        s = _think()
        # 2 tokens over a ramp of 4 -> half of max_bias.
        past = torch.tensor([[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]])
        self.assertAlmostEqual(s._bias(past), 8.0, places = 5)
        longer = torch.tensor([[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]])
        self.assertGreater(s._bias(longer), s._bias(past))
        way_over = torch.tensor([[1, 2, 3, 4, 5] + [7] * 20])
        self.assertAlmostEqual(s._bias(way_over), 16.0, places = 5)


class _State:
    def __init__(self, past, logits):
        self.past_ids = past
        self.logits = logits
        self.dim = logits.shape[-1]


class TestThinkFreq(unittest.TestCase):
    def test_think_freq_lowers_repeated_tokens_only_while_think_is_open(self):
        s = SS_ThinkExit(
            5, prompt_len = 2, budget = 100, ramp = 4, max_bias = 16,
            think_freq = 1.0, think_freq_window = 32)
        open_state = _State(torch.tensor([[1, 2] + [7] * 16 + [4]]), torch.zeros(1, 12))
        s._penalize_open_think(open_state)
        self.assertEqual(open_state.logits[0, 7].item(), -16)
        self.assertEqual(open_state.logits[0, 5].item(), 0)
        closed = _State(torch.tensor([[1, 5, 7, 7, 7, 7]]), torch.zeros(1, 12))
        s._penalize_open_think(closed)
        self.assertEqual(torch.count_nonzero(closed.logits).item(), 0)

    def test_close_token_is_not_frequency_penalized(self):
        s = SS_ThinkExit(
            7, prompt_len = 1, budget = 100, ramp = 4, max_bias = 16,
            think_freq = 1.0, think_freq_window = 32)
        state = _State(torch.tensor([[1] + [7] * 20]), torch.zeros(1, 12))
        s._penalize_open_think(state)
        # token 7 is the close id, so its penalty is cleared even though it repeated
        self.assertEqual(state.logits[0, 7].item(), 0)


class TestDRY(unittest.TestCase):
    def _logits(self, tokens, allowed = 6):
        past = torch.tensor([tokens])
        logits = torch.zeros(1, 64)
        state = SamplingState(
            rand_u32 = 1, bsz = 1, dim = 64,
            in_logits = logits, past_ids = past, state = SS.INIT)
        SS_DRY(0.2, 1.75, allowed, 4096, dry_sequence_breakers = None).run(state)
        return state.logits

    def test_exact_repeat_of_length_6_is_not_penalized(self):
        # Ending copies a 6-token span. The token that would continue it (42)
        # is not down-weighted.
        tokens = [9, 1, 2, 3, 4, 5, 6, 42, 1, 2, 3, 4, 5, 6]
        logits = self._logits(tokens)
        self.assertEqual(logits[0, 42].item(), 0.0)

    def test_longer_exact_repeat_is_penalized(self):
        tokens = [9, 1, 2, 3, 4, 5, 6, 7, 42, 1, 2, 3, 4, 5, 6, 7]
        logits = self._logits(tokens)
        # match length 7, allowed 6 -> multiplier * base ** 1
        self.assertAlmostEqual(logits[0, 42].item(), -0.2 * 1.75, places = 5)
        self.assertEqual(logits[0, 3].item(), 0.0)


class TestServerUsesThoseSteps(unittest.TestCase):
    def test_build_sampler_uses_dry_and_think_exit(self):
        sampler = so.build_sampler(
            0.0, 20, 0.95,
            1.0, 0.0, 512, 0.0,
            0.2, 1.75, 6, 4096, None,
            True, 99, 5,
            4096, 1536, 16.0, 0.2, 256,
        )
        kinds = [type(step) for step in sampler.steps]
        self.assertIn(SS_DRY, kinds)
        self.assertIn(SS_ThinkExit, kinds)
        dry = next(step for step in sampler.steps if isinstance(step, SS_DRY))
        self.assertEqual(dry.dry_allowed_length, 6)
        self.assertEqual(dry.dry_multiplier, 0.2)
        think = next(step for step in sampler.steps if isinstance(step, SS_ThinkExit))
        self.assertEqual(think.budget, 4096)
        self.assertEqual(think.ramp, 1536)
        self.assertEqual(think.think_freq, 0.2)

    def test_omitted_max_tokens_is_32768(self):
        self.assertEqual(so.DEFAULT_MAX_TOKENS, 32768)
        req, err = so.parse_request({"messages": [{"role": "user", "content": "hi"}]})
        self.assertIsNone(err)
        self.assertEqual(req["max_tokens"], 32768)

    def test_explicit_frequency_penalty_zero_stays_zero(self):
        saved = so.SERVER_DEFAULTS["freq_penalty"]
        so.SERVER_DEFAULTS["freq_penalty"] = 0.4
        try:
            req, err = so.parse_request({
                "messages": [{"role": "user", "content": "hi"}],
                "frequency_penalty": 0,
            })
            self.assertIsNone(err)
            self.assertEqual(req["freq_penalty"], 0.0)
            omitted, err = so.parse_request(
                {"messages": [{"role": "user", "content": "hi"}]})
            self.assertEqual(omitted["freq_penalty"], 0.4)
        finally:
            so.SERVER_DEFAULTS["freq_penalty"] = saved

    def test_repetition_penalty_zero_leaves_logits_unchanged(self):
        # Client 0 is passed into the sampler the server forwards. The kernel
        # divides by rep_p, so a live penalty would turn the repeated positive
        # logit into +inf and steal argmax from the real maximum.
        req, err = so.parse_request({
            "messages": [{"role": "user", "content": "hi"}],
            "repetition_penalty": 0,
        })
        self.assertIsNone(err)
        self.assertEqual(req["rep_penalty"], 0.0)
        self.assertTrue(torch.cuda.is_available())
        # Default fusion stays on: this is the stack build_sampler hands the server.
        sampler = so.build_sampler(
            0.0, 20, 0.95,
            req["rep_penalty"], 0.0, 512, 0.0,
            0.0, 1.75, 6, 4096, None,
            False, None, 3,
            0, 1536, 16.0, 0.0, 256,
        )
        reps = [step for step in sampler.steps if isinstance(step, SS_RepP)]
        self.assertEqual([type(step) for step in sampler.steps[:1]], [SS_RepP])
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0].rep_p, 0.0)
        calls = []
        orig = sampler_custom.ext.apply_rep_pens
        def spy(*args, **kwargs):
            calls.append(1)
            return orig(*args, **kwargs)
        sampler_custom.ext.apply_rep_pens = spy
        try:
            base = torch.tensor([[1.0, 5.0, -0.25]], device = "cuda")
            # Token 0 is repeated. v/0 would make its logit +inf and beat token 1.
            past = torch.tensor([[0, 0, 0]], device = "cuda")
            state = sampler.forward(base.clone(), sequence_ids = past, return_state = True)
        finally:
            sampler_custom.ext.apply_rep_pens = orig
        self.assertEqual(calls, [])
        self.assertTrue(torch.allclose(state.logits.float().cpu(), base.float().cpu()))
        self.assertFalse(torch.isinf(state.logits).any().item())
        self.assertEqual(int(state.sample.view(-1)[0]), 1)

    def test_explicit_dry_zero_turns_it_off(self):
        req, err = so.parse_request({
            "messages": [{"role": "user", "content": "hi"}],
            "dry_multiplier": 0,
        })
        self.assertIsNone(err)
        self.assertEqual(req["dry_multiplier"], 0.0)

    def test_default_loop_window_is_off(self):
        self.assertEqual(so.SERVER_DEFAULTS["loop_window"], 0)
        self.assertIsNone(so.loop_stop_from(0, 3))
        self.assertEqual(so.loop_stop_from(8, 3), (8, 3))

    def test_acceptance_length_definition(self):
        # new_tokens / max(1, new_tokens - accepted)
        self.assertAlmostEqual(so.draft_acceptance(10, 8), 5.0)
        self.assertAlmostEqual(so.draft_acceptance(4, 4), 4.0)
        self.assertEqual(so.draft_acceptance(0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
