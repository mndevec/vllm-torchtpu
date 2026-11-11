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
"""DCP (Decode Context Parallelism) attention for MLA: the head gather and
the log-sum-exp merge that bracket the sharded attention call.

The MLA counterpart of `cp_attention.py`.

Every collective runs on tensors already in torch's hands, outside
the exported op.

Each DCP rank stores a shard of the KV cache, so it can only produce a
*partial* attention result.

  1. `all_gather_heads` widens this rank's query head slice to the whole
     group's heads, so every rank's heads are attended against every rank's
     KV shard.
  2. the kernel runs against this rank's shard alone, returning a partial
     output and its log-sum-exp.
  3. `merge_lse_partials_scatter_heads` folds the partials into the exact
     whole-context result and narrows the head axis back to this rank's own.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.collectives import all_to_all
from vllm_torchtpu.distributed.dcp import get_dcp_group
from vllm_torchtpu.distributed.dcp_comm import cp_lse_ag_out_rs


def all_gather_heads(q: torch.Tensor) -> torch.Tensor:
    """Widen a rank's query head slice to the whole DCP group's heads.

    DCP doesn't add a new parallel dimension—it borrows from TP. It
    reallocates dcp ways from heads to sequence positions, reducing
    head sharding from tp to tp // dcp.

    Args:
      q: `[num_tokens, tp_local_num_heads, dim]`, this rank's head slice.

    Returns:
      `[num_tokens, tp_local_num_heads * dcp_size, dim]`, identical
      on every rank. Returned unchanged without a DCP group.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return q
    return group.all_gather(q.contiguous(), dim=1)


def all_gather_q_nope_pe_heads(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused all-gather of `(ql_nope, q_pe)` across DCP ranks in a single call."""
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return ql_nope, q_pe
    q_dcp = group.all_gather(torch.cat([ql_nope, q_pe], dim=-1), dim=1)
    return (
        q_dcp[..., :kv_lora_rank].contiguous(),
        q_dcp[..., kv_lora_rank:].contiguous(),
    )


def merge_lse_partials_scatter_heads(
    output: torch.Tensor, lse: torch.Tensor, is_backbone: bool = True
) -> torch.Tensor:
    """Combine this rank's partial attention output with its peers'.

    With per-rank normalized outputs `o_r` and `lse_r = m_r + log(l_r)`, the
    exact whole-context result is `o = sum_r softmax_r(lse) * o_r`. Correctness
    does not depend on the order the ranks are combined in.

    Two collectives:

      1. all-gather the log-sum-exps. They are `[num_tokens, num_heads]` f32 --
         Leaves every rank holding every `lse_r`.
      2. reduce-scatter the weighted outputs

    With TPU_DCP_LSE_MERGE_A2A set, the backbone's steps of at most
    TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS tokens instead take
    `merge_lse_partials_all_to_all`, which gives the same result.

    Args:
      output: `[num_tokens, tp_local_num_heads * dcp_size, kv_lora_rank]`,
        this rank's partial.
      lse: `[num_tokens, tp_local_num_heads * dcp_size]`, the matching log-sum-exps.
        Entries are `-inf` for tokens this rank owns none of the top-k for;
        such a rank was still fed a dummy key (the kernel requires
        `kv_len >= 1`) and its output must not be weighted in.
      is_backbone: True if the caller is the backbone, the only model that is
        traced separately per token bucket. Only then may the merge compare
        the token count to TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS, since that
        comparison is fixed at trace time. Pass False from the MTP drafter,
        which is traced once for every bucket: it always takes the
        reduce-scatter, exactly as with TPU_DCP_LSE_MERGE_A2A unset.

    Returns:
      `[num_tokens, tp_local_num_heads, kv_lora_rank]` in `output.dtype` --
      this rank's head slice of the whole-context result.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return output

    # This `if` runs while tracing, so its answer is baked into the graph.
    # Only the backbone is traced once per bucket group, so only it can pick
    # by size: the all-to-all wins at decode sizes, the reduce-scatter above
    # because it sums in the network. The MTP drafter has one graph for every
    # bucket; comparing its token count would pin that graph to the first
    # size traced. `is_backbone` is checked before the token count, so the
    # drafter never compares it.
    if (
        envs.TPU_DCP_LSE_MERGE_A2A
        and is_backbone
        and output.shape[0] <= envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS
    ):
        return merge_lse_partials_all_to_all(output, lse, group)

    rs_dtype = output.dtype if envs.TPU_DCP_RS_OUT_PRECISION and is_backbone else None
    return cp_lse_ag_out_rs(output, lse, group, rs_dtype=rs_dtype)


def merge_lse_partials_all_to_all(
    output: torch.Tensor, lse: torch.Tensor, group: Any
) -> torch.Tensor:
    """Same result as the reduce-scatter merge, via all-to-all.

    Each rank sends head block `d` of its partial and lse to rank `d`, then
    merges its own heads locally in f32. Faster for small token counts.
    """
    world = int(group.world_size)
    num_tokens, num_heads, vdim = output.shape
    local = num_heads // world
    # Destination-major: [world, T, local, V] and [world, T, local].
    o_send = output.reshape(num_tokens, world, local, vdim).transpose(0, 1)
    l_send = lse.to(torch.float32).reshape(num_tokens, world, local).transpose(0, 1)
    if envs.TPU_DCP_LSE_MERGE_A2A_PACK and o_send.element_size() == 2:
        o_recv, l_recv = _all_to_all_packed(o_send, l_send, group)
    else:
        o_recv = all_to_all(o_send, group)
        l_recv = all_to_all(l_send, group)
    return _merge_received(o_recv, l_recv).to(output.dtype)


def _all_to_all_packed(
    o_send: torch.Tensor,
    l_send: torch.Tensor,
    group: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(all_to_all(o_send), all_to_all(l_send))` in ONE all-to-all
    (`TPU_DCP_LSE_MERGE_A2A_PACK`).

    `o_send` is `[world, T, local, V]` in a 2-byte dtype and `l_send` the f32
    `[world, T, local]`. The exchange carries int16 words: the partial
    reinterpreted as is, then each destination's `T * local` lse bit patterns
    split into a block of high halves and a block of low halves (integer
    shifts and masks, so no lane interleave), zero padded to whole `V`-wide
    rows after its `T * local` partial rows. The words stay integers until
    they are unpacked: a float concat may be lowered to a select/max that
    canonicalizes the NaN bit patterns half an f32 can take. Both halves come
    back bit for bit what the two separate exchanges return; only the f32 lse's
    own collective (latency-bound at a few KB) is gone, for
    `ceil(2 * T * local / V)` extra rows on the partial's.
    """
    world, tokens, local, vdim = o_send.shape
    rows = tokens * local
    words = rows * 2
    extra = -(-words // vdim)
    o_words = o_send.reshape(world, rows, vdim).view(torch.int16)
    bits = l_send.contiguous().view(torch.int32).reshape(world, rows)
    hi = (bits >> 16).to(torch.int16)
    # The low half as a signed 16-bit value, so the narrowing is exact.
    lo = (((bits & 0xFFFF) ^ 0x8000) - 0x8000).to(torch.int16)
    l_words = torch.nn.functional.pad(
        torch.cat([hi, lo], dim=1), (0, extra * vdim - words)
    )
    send = torch.cat([o_words, l_words.reshape(world, extra, vdim)], dim=1)
    recv = all_to_all(send, group)
    o_recv = recv[:, :rows].reshape(world, tokens, local, vdim).view(o_send.dtype)
    l_recv_words = recv[:, rows:].reshape(world, extra * vdim)
    hi = l_recv_words[:, :rows].to(torch.int32)
    lo = l_recv_words[:, rows:words].to(torch.int32)
    l_recv = (
        ((hi << 16) | (lo & 0xFFFF)).view(torch.float32).reshape(world, tokens, local)
    )
    return o_recv, l_recv


def _merge_received(o_recv: torch.Tensor, l_recv: torch.Tensor) -> torch.Tensor:
    """`[R, T, h, V]` partials and `[R, T, h]` lse -> the f32 `[T, h, V]` merge."""
    finite = torch.isfinite(l_recv)
    shift = torch.where(finite, l_recv, torch.full_like(l_recv, float("-inf"))).amax(
        dim=0
    )
    shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
    weights = torch.where(
        finite, torch.exp(l_recv - shift.unsqueeze(0)), torch.zeros_like(l_recv)
    )
    denom = weights.sum(dim=0)
    weights = weights / torch.where(denom > 0, denom, torch.ones_like(denom))
    # The compiler lowers multiply-then-sum to a dot that rounds f32 operands to
    # bf16. Splitting the weights into a bf16 part and a residual keeps it exact.
    o_f32 = o_recv.to(torch.float32)
    w_hi = weights.to(torch.bfloat16).to(torch.float32)
    w_lo = weights - w_hi
    return (o_f32 * w_hi.unsqueeze(-1)).sum(dim=0) + (o_f32 * w_lo.unsqueeze(-1)).sum(
        dim=0
    )
