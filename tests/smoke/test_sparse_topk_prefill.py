# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Compact prefill selection, including exact refinement and carrier reuse."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from fmha_sm100 import sparse_topk_select_prefill


@pytest.mark.parametrize(
    "pages,mode",
    [
        (127, "unique"),
        (128, "unique"),
        (129, "unique"),
        (129, "tie"),
        (1600, "unique"),
        (1600, "cluster"),
        (1600, "low_bits"),
        (1600, "cross_bin"),
        (10000, "cluster"),
        (1600, "zero"),
    ],
)
@pytest.mark.parametrize("force_begin,force_end", [(0, 0), (2, 3)])
def test_prefill_compact_selection(pages, mode, force_begin, force_end):
    width = pages + 31
    lengths = [0, 1, 15, 16, 17, 128, pages]
    tokens, heads = len(lengths), 2
    scores = torch.empty((tokens, heads, width), device="cuda")
    valid = torch.empty(tokens, device="cuda", dtype=torch.int32)
    sentinel = 0x13131313
    storage = torch.full(
        (heads, 2 * tokens + 3, 41), sentinel, device="cuda", dtype=torch.int32
    )
    carrier = storage[:, 2 : 2 + 2 * tokens : 2, :33]
    guard = torch.ones_like(storage, dtype=torch.bool)
    guard[:, 2 : 2 + 2 * tokens : 2, :33] = False
    for reuse in range(2):
        current_lengths = lengths if reuse == 0 else lengths[::-1]
        source = torch.full((tokens, heads, width), float("nan"))
        for token, count in enumerate(current_lengths):
            if mode == "unique":
                values = torch.arange(count, dtype=torch.float32) - count / 2
            elif mode == "cluster":
                values = (0x3F070000 + torch.arange(count, dtype=torch.int32)).view(
                    torch.float32
                )
            elif mode == "low_bits":
                # More than 128 entries share all but the final two fp32 bits.
                values = (0x3F070000 + torch.arange(count, dtype=torch.int32) % 4).view(
                    torch.float32
                )
            elif mode == "zero":
                values = torch.zeros(count)
                values[::2] = -0.0
            elif mode == "cross_bin":
                values = torch.full((count,), 0.51)
                values[-min(7, count) :] = 0.70
            else:
                values = torch.full((count,), 0.5)
            if reuse:
                values = values.flip(0)
            values[: min(force_begin, count)] = -100.0
            if force_end:
                values[max(0, count - force_end) :] = -100.0
            source[token] = torch.cat(
                (values, torch.full((width - count,), float("nan")))
            )
        scores.copy_(source)
        valid.copy_(torch.tensor(current_lengths, dtype=torch.int32))
        sparse_topk_select_prefill(
            scores,
            valid,
            carrier,
            force_begin_blocks=force_begin,
            force_end_blocks=force_end,
        )
        result = carrier.cpu().permute(1, 0, 2)
        assert torch.all(storage[guard] == sentinel)
        assert torch.all(result[..., 32] == 1)
        for token, count in enumerate(current_lengths):
            forced = set(range(min(force_begin, count))) | set(
                range(max(0, count - force_end), count)
            )
            ordinary = [i for i in range(count) if i not in forced]
            quota = min(16, count) - len(forced)
            expected_scores = (
                source[token, 0, ordinary].topk(quota).values.sort().values
            )
            for head in range(heads):
                ids = result[token, head, 16:32].tolist()
                live = ids[: min(16, count)]
                assert live == sorted(set(live))
                assert all(0 <= i < count for i in live)
                assert ids[min(16, count) :] == [-1] * (16 - min(16, count))
                assert forced.issubset(live)
                selected = [i for i in live if i not in forced]
                torch.testing.assert_close(
                    source[token, head, selected].sort().values,
                    expected_scores,
                    rtol=0,
                    atol=0,
                )
                expected_bits = torch.full((16,), -float("inf")).view(torch.int32)
                if live:
                    expected_bits[: len(live)] = source[token, head, live].view(
                        torch.int32
                    )
                assert torch.equal(result[token, head, :16], expected_bits)
