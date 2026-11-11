# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decode-size TP all-reduces through the Pallas in-kernel all-reduce
(TPU_PALLAS_TP_ALL_REDUCE).

The TP all-reduce of a bf16 [T, h] partial after a RowParallelLinear (MLA
o_proj, a dense MLP's down_proj) and after an MoE layer (the runner's final
all-reduce of shared + routed) is latency-bound at decode token counts: XLA's
all-reduce pays a fixed launch and barrier cost the payload does not need.
`kernels/experimental/moe_tp_decode/all_reduce.py` is a one-shot
reduce-scatter + all-gather over remote DMAs that the fused moe_tp_decode
kernel already uses as its epilogue; `standalone` wraps it as a pallas_call.

An armed layer replaces its `tensor_model_parallel_all_reduce` with one torch
op over the TP group's mesh (the moe_tp_decode mesh and rank operand). The
token count is static inside the op (torch_tpu lowers one program per compile
size), so the op itself picks the path: the Pallas all-reduce when
T <= TPU_PALLAS_TP_ALL_REDUCE_MAX_TOKENS and `all_reduce.supported(n, T, h)`
(T a whole number of bf16 row tiles, h / n lane-aligned), else `lax.psum`.
Every row is valid (n_valid = T, the kernel's full path), so padded rows are
reduced exactly as XLA reduces them. Sums are f32 in mesh order rounded once
to bf16, so they may differ from XLA's by bf16 rounding order only.

`arm` runs once at the end of TPUModelRunner.load_model. It is collective
over the TP group: every rank must arm the same layers (a rank that skips the
kernel would leave its peers waiting in its barrier), so the ranks compare an
arming signature and raise on a mismatch, then run the armed op once on
inputs whose sums are exact at every token count it serves with the kernel
and one it serves with the psum, and raise if any rank's result is wrong.
"""

import faulthandler
import hashlib
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Attributes an armed module carries: the op, and this rank's mesh index as a
# non-persistent buffer (data, so every rank traces the same program).
OP_ATTR = "_pallas_tp_ar_op"
RANK_BUFFER = "_pallas_tp_ar_rank"

_OPS: dict[tuple, Any] = {}
_SELFTEST_TIMEOUT_S = 300


def armed_op(module: torch.nn.Module) -> Any | None:
    """The armed all-reduce op of `module`, or None (unarmed: upstream path)."""
    return getattr(module, OP_ATTR, None)


def use_pallas(n_ranks: int, rows: int, hidden: int, max_tokens: int, info) -> bool:
    """Whether the op runs the Pallas all-reduce for a static [rows, hidden]
    partial over n_ranks (else lax.psum)."""
    return rows <= max_tokens and all_reduce.supported(n_ranks, rows, hidden, info)


def op_body(n_ranks: int, axis: str, max_tokens: int):
    """Per-rank body under shard_map over `axis`: (x bf16 [T, h], rank int32
    [1, 1]) -> sum over the axis of x, [T, h]."""

    def body(x, rank):
        rows, hidden = x.shape  # static: one program per compile size
        info = pltpu.get_tpu_info()
        if x.dtype == jnp.bfloat16 and use_pallas(
            n_ranks, rows, hidden, max_tokens, info
        ):
            spec = all_reduce.make_spec(n_ranks, rows, hidden)
            meta = jnp.stack(
                [jnp.int32(rows), rank.reshape(()).astype(jnp.int32)]
            ).astype(jnp.int32)
            return all_reduce.standalone(spec)(meta, x)
        return lax.psum(x, axis)

    return body


def get_op(mesh, max_tokens: int):
    """One jax_op per (mesh, threshold); built at arming, never in a trace."""
    key = (mesh, max_tokens)
    op = _OPS.get(key)
    if op is not None:
        return op
    axis = mesh.axis_names[0]
    specs = (P(axis), P(axis))
    sharded = jax.shard_map(
        op_body(int(mesh.size), axis, max_tokens),
        mesh=mesh,
        in_specs=specs,
        out_specs=P(axis),
        check_vma=False,
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        args = (x, rank)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    name = "pallas::tp_all_reduce" + (f"_{len(_OPS)}" if _OPS else "")
    # PartitionSpecs, as in moe_tp_decode._get_op.
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        name, fn, mesh=mesh, input_partition_specs=specs, output_partition_specs=P(axis)
    )
    op.register_fake(lambda x, *args, **kwargs: torch.empty_like(x))
    _OPS[key] = op
    return op


def _rank_holder(module: torch.nn.Module) -> torch.nn.Module:
    """The module that holds the rank buffer: an MoE runner's routed experts
    (where moe_tp_decode keeps its rank buffer), else the module itself."""
    experts = getattr(module, "routed_experts", None)
    return module if experts is None else experts


def all_reduce_tp(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    """`x` all-reduced over the TP group by `module`'s armed op, or None when
    the module is unarmed or `x` is not a bf16 matrix (the caller then runs
    tensor_model_parallel_all_reduce). Only static properties are read."""
    op = armed_op(module)
    if op is None or x.dtype != torch.bfloat16 or x.dim() != 2:
        return None
    return op(x.contiguous(), getattr(_rank_holder(module), RANK_BUFFER))


def _targets(model: torch.nn.Module | None, n_ranks: int, seen: set[int]):
    """(module, hidden) for every layer whose TP all-reduce this arms."""
    from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner
    from vllm_torchtpu.layers.adapter.linear import TpuRowParallelLinear

    if model is None:
        return []
    out = []
    for m in model.modules():
        if id(m) in seen:
            continue
        if isinstance(m, TpuRowParallelLinear):
            if m.reduce_results and m.tp_size == n_ranks:
                seen.add(id(m))
                out.append((m, int(m.output_size)))
        elif isinstance(m, VllmMoERunner):
            cfg = m.moe_config
            if (
                not cfg.is_sequence_parallel
                and not cfg.skip_final_all_reduce
                and (cfg.tp_size > 1 or cfg.ep_size > 1)
            ):
                seen.add(id(m))
                out.append((m, int(cfg.hidden_dim)))
    return out


def _selftest(op, rank: torch.Tensor, mesh_index: int, n_ranks: int, cases, device):
    """Run the armed op on selftest_inputs for every (rows, hidden) in
    `cases`; returns this rank's failing cases. Collective."""
    failures = []
    faulthandler.dump_traceback_later(_SELFTEST_TIMEOUT_S, exit=True)
    try:
        for rows, hidden in cases:
            # Lane blocks of the real lane count, so a misplaced chunk shows.
            info = pltpu.get_tpu_info()
            spec = all_reduce.AllReduceSpec(
                n_ranks, rows, hidden, info.get_sublane_tiling(jnp.bfloat16),
                info.num_lanes,
            )
            x = all_reduce.selftest_inputs(spec)
            mine = torch.from_numpy(x[mesh_index]).to(torch.bfloat16).to(device)
            out = op(mine, rank).to(torch.float32).cpu().numpy()
            if not all_reduce.selftest_ok(spec, x, out, rows):
                failures.append((rows, hidden))
    finally:
        faulthandler.cancel_dump_traceback_later()
    return failures


def arm(model: torch.nn.Module, draft_model: torch.nn.Module | None = None) -> int:
    """Arm the TP all-reduce of every RowParallelLinear (reduce_results) and
    MoE runner (final all-reduce) of `model` and `draft_model`; returns how
    many were armed. Collective over the TP group."""
    # Reuse moe_tp_decode's mesh, rank tensor and TP-group gather: same group,
    # same kernel protocol, same rank operand.
    from vllm_torchtpu.layers.adapter import moe_tp_decode
    from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner

    max_tokens = envs.TPU_PALLAS_TP_ALL_REDUCE_MAX_TOKENS
    topo = moe_tp_decode._topology()  # collective: first, on every rank
    if isinstance(topo, str):
        plan, reason, n_ranks, mesh_index = [], topo, 0, -1
    else:
        n_ranks, mesh_index = int(topo.mesh.size), topo.mesh_index
        seen: set[int] = set()
        plan = _targets(model, n_ranks, seen) + _targets(draft_model, n_ranks, seen)
        reason = "none" if plan else "no TP all-reduce to arm"
    info = pltpu.get_tpu_info()
    tile = info.get_sublane_tiling(jnp.bfloat16)
    hiddens = sorted({h for _, h in plan})
    kernel_cases = [
        (m, h)
        for h in hiddens
        for m in range(tile, max_tokens + 1, tile)
        if use_pallas(n_ranks, m, h, max_tokens, info)
    ]
    sig = hashlib.sha256(
        repr(
            (
                [(type(m).__name__, h) for m, h in plan],
                max_tokens,
                kernel_cases,
            )
        ).encode()
    ).hexdigest()[:16]
    moe_tp_decode._check_ranks_agree(
        dict(sig=sig, mesh_index=mesh_index, armed=len(plan))
    )
    if not plan:
        logger.warning("pallas_tp_all_reduce: armed nothing (%s)", reason)
        return 0
    assert not isinstance(topo, str)
    if not kernel_cases:
        logger.warning(
            "pallas_tp_all_reduce: no token count <= %d admits the kernel at "
            "hidden sizes %s over %d ranks; arming nothing",
            max_tokens,
            hiddens,
            n_ranks,
        )
        return 0
    device = next(
        (p.device for m, _ in plan for p in m.parameters(recurse=True)),
        torch.device(torch.get_default_device()),
    )
    op = get_op(topo.mesh, max_tokens)
    rank = moe_tp_decode._rank_tensor(device, mesh_index)
    # Every kernel case, plus one psum case per hidden size above the
    # threshold, so a broken fallback fails here too.
    psum_cases = [(max_tokens + tile - max_tokens % tile, h) for h in hiddens]
    failures = _selftest(op, rank, mesh_index, n_ranks, kernel_cases + psum_cases, device)
    seen_failures = moe_tp_decode._gather(dict(mesh_index=mesh_index, failures=failures))
    if any(s["failures"] for s in seen_failures):
        raise RuntimeError(
            "pallas_tp_all_reduce: the all-reduce self-test failed (per rank: "
            f"mesh index, failing (rows, hidden)): {seen_failures}"
        )
    for module, _ in plan:
        _rank_holder(module).register_buffer(RANK_BUFFER, rank, persistent=False)
        setattr(module, OP_ATTR, op)
    n_moe = sum(1 for m, _ in plan if isinstance(m, VllmMoERunner))
    logger.info(
        "pallas_tp_all_reduce: armed %d linear and %d MoE all-reduces (mesh "
        "index %d of %d, kernel token counts %s, hidden %s, sig %s)",
        len(plan) - n_moe,
        n_moe,
        mesh_index,
        n_ranks,
        sorted({m for m, _ in kernel_cases}),
        hiddens,
        sig,
    )
    return len(plan)
