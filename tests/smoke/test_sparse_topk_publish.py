# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Decode publish TopK: exactly the merge's order (score descending with
forced pages as +inf, NaN as -inf and -0 as +0, then the lower column), in
tagged and legacy rows; peer windows are on one GPU."""

import pytest
import torch

from fmha_sm100.jit import get_sparse_topk_module

INF_BITS = torch.tensor([-float("inf")]).view(torch.int32).item()


def _local_count(count, scale, offset):
    return 0 if count <= offset else (count - 1 - offset) // scale + 1


def _expected_columns(row, valid, scale, offset, force_begin, force_end):
    """Selected local columns of one row, ascending, padded with -1."""
    width = row.numel()
    global_valid = min(max(valid, 0), width * scale)
    local_valid = _local_count(global_valid, scale, offset)
    end = min(local_valid, width)
    if end <= 16:
        return list(range(end)) + [-1] * (16 - end)
    begin = _local_count(force_begin, scale, offset)
    end_start = _local_count(
        global_valid - force_end if force_end <= global_valid else 0, scale, offset
    )
    values = row[:end].tolist()

    def key(column):
        if column < begin or end_start <= column < local_valid:
            value = float("inf")
        else:
            value = values[column]
            value = -float("inf") if value != value else value + 0.0
        return (value, -column)

    return sorted(sorted(range(end), key=key, reverse=True)[:16])


def _row_values(mode, width, generator):
    if mode == "random":
        return torch.randn(width, generator=generator)
    if mode == "ties":
        return torch.randint(0, 3, (width,), generator=generator).float()
    if mode == "zeros":
        values = torch.zeros(width)
        values[1::2] = -0.0
        values[::7] = torch.rand(len(values[::7]), generator=generator)
        return values
    if mode == "special":
        # +inf ties forced pages; NaN ranks as -inf.
        values = torch.randn(width, generator=generator)
        pick = torch.rand(width, generator=generator)
        values[pick < 0.05] = float("inf")
        values[(pick >= 0.05) & (pick < 0.1)] = -float("inf")
        values[(pick >= 0.1) & (pick < 0.15)] = float("nan")
        return values
    # "lanes": the largest scores sit in two lanes' columns, which overflows
    # the survivor buffer for wide rows and takes the exact fallback.
    values = torch.rand(width, generator=generator)
    columns = torch.arange(width)
    lanes = (columns % 32 == 0) | (columns % 32 == 5)
    values[lanes] = 10.0 + columns[lanes].float() / width
    return values


def _publish_and_check(
    source, lengths, world, hpr, scale, offset, force_begin, force_end, tagged,
    pdl=False, steps=1,
):
    """Publish `source` (one list entry per step) from every rank into the same
    slot and check each step's rows exactly."""
    heads = world * hpr
    tokens = len(lengths)
    capacity = tokens + 2
    slot = 1
    if tagged:
        windows = [
            torch.zeros((3, world, hpr, capacity, 32), device="cuda", dtype=torch.int64)
            for _ in range(world)
        ]
        # Each rank counts its own publishes.
        generations = [
            torch.zeros((3, heads, capacity), device="cuda", dtype=torch.int32)
            for _ in range(world)
        ]
    else:
        windows = [
            torch.zeros((3, world, hpr, capacity, 33), device="cuda", dtype=torch.int32)
            for _ in range(world)
        ]
        generations = [None] * world
    bases = torch.tensor([w.data_ptr() for w in windows], device="cuda", dtype=torch.int64)
    valid = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    errors = torch.zeros(1, device="cuda", dtype=torch.int32)
    publish = get_sparse_topk_module().sparse_topk_select_publish
    for step in range(steps):
        scores = source[step].cuda()
        if not tagged:
            # The owner consumes a legacy row by clearing its ready flag.
            for window in windows:
                window[..., 32] = 0
        for rank in range(world):
            publish(
                scores,
                valid,
                bases,
                errors,
                slot,
                rank,
                world,
                hpr,
                capacity,
                force_begin,
                force_end,
                1 << 20,
                pdl,
                generations[rank],
                scale,
                offset,
                torch.cuda.current_stream().cuda_stream,
            )
        torch.cuda.synchronize()
        assert errors.item() == 0
        rows = [w.cpu() for w in windows]
        for rank in range(world):
            for token, count in enumerate(lengths):
                for head in range(heads):
                    row = rows[head // hpr][slot, rank, head % hpr, token]
                    if tagged:
                        assert torch.all((row >> 32) == step + 1)
                        words = (row & 0xFFFFFFFF).to(torch.int64)
                        words = torch.where(words >= 2**31, words - 2**32, words).to(
                            torch.int32
                        )
                    else:
                        assert row[32].item() == 1
                        words = row[:32]
                    columns = _expected_columns(
                        source[step][token, head], count, scale, offset, force_begin,
                        force_end,
                    )
                    expected_ids = [c * scale + offset if c >= 0 else -1 for c in columns]
                    assert words[16:32].tolist() == expected_ids, (step, rank, token, head)
                    expected_bits = [
                        source[step][token, head, c].view(torch.int32).item()
                        if c >= 0 else INF_BITS
                        for c in columns
                    ]
                    assert words[:16].tolist() == expected_bits, (step, rank, token, head)
        # Only this slot's rows of real tokens were written.
        for window in rows:
            assert torch.count_nonzero(window[[0, 2]]) == 0
            assert torch.count_nonzero(window[slot, :, :, tokens:]) == 0


def _source(mode, lengths, heads, width, scale, offset, force_end, generator):
    source = torch.full((len(lengths), heads, width), float("nan"))
    for token, count in enumerate(lengths):
        local = min(_local_count(min(count, width * scale), scale, offset), width)
        for head in range(heads):
            source[token, head, :local] = _row_values(mode, local, generator)
        # Forced pages may hold stale NaN; they are selected regardless.
        if local > 16 and force_end:
            source[token, :, local - 1] = float("nan")
    return source


@pytest.mark.parametrize("width", [17, 200, 500, 700, 1000, 1024, 1025, 1536, 5000])
@pytest.mark.parametrize("mode", ["random", "ties", "zeros", "special", "lanes"])
@pytest.mark.parametrize("scale,offset", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("force_begin,force_end", [(0, 1), (2, 3)])
@pytest.mark.parametrize("tagged", [True, False])
def test_publish_selects_exact_topk(
    width, mode, scale, offset, force_begin, force_end, tagged
):
    world, hpr = 2, 2
    lengths = [0, 1, 16, 17, 33, width * scale // 2, width * scale, width * scale + 5]
    generator = torch.Generator().manual_seed(width * 7 + scale + offset)
    source = _source(
        mode, lengths, world * hpr, width, scale, offset, force_end, generator
    )
    _publish_and_check(
        [source], lengths, world, hpr, scale, offset, force_begin, force_end, tagged
    )


@pytest.mark.parametrize("world,hpr", [(2, 1), (2, 2), (4, 1)])
@pytest.mark.parametrize("tagged", [True, False])
@pytest.mark.parametrize("pdl", [False, True])
def test_publish_reuses_slots(world, hpr, tagged, pdl):
    """Two steps into one slot (tagged generations 1 then 2, legacy rows
    consumed in between); nine tokens leave the last CTA partly empty when
    there are two heads."""
    width, scale, offset = 1300, 1, 0
    lengths = [1300, 17, 0, 640, 1299, 16, 900, 1300, 33]
    generator = torch.Generator().manual_seed(world * 10 + hpr)
    source = [
        _source("random", lengths, world * hpr, width, scale, offset, 1, generator)
        for _ in range(2)
    ]
    _publish_and_check(
        source, lengths, world, hpr, scale, offset, 0, 1, tagged, pdl=pdl, steps=2
    )
