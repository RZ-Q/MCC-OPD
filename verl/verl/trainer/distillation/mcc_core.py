# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
#
# Modified for MCC-OPD: standalone union-partition and mirror-calibration interface.

"""MCC-OPD tensor-level core, extracted from the training implementation.

Model forwarding, response-position alignment, and distributed execution are
caller responsibilities. See README.md for the input contract and verl reduction.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class UnionGroup:
    """Response positions with the same deduplicated support size."""

    positions: torch.Tensor
    token_ids: torch.Tensor


@torch.no_grad()
def build_union_groups(student_ids, teacher_ids):
    """Union of student and full-context teacher top-K IDs, shaped [1, T, K]."""
    overlap = (teacher_ids.unsqueeze(-1) == student_ids.unsqueeze(-2)).any(dim=-1)
    ids = torch.cat((student_ids, teacher_ids), dim=-1)
    valid = torch.cat((torch.ones_like(student_ids, dtype=torch.bool), ~overlap), dim=-1)
    order = torch.arange(ids.shape[-1], device=ids.device).expand_as(ids)
    order = torch.where(valid, order, ids.shape[-1]).argsort(dim=-1)
    lengths = valid.sum(dim=-1)[0]
    groups = []
    for size in lengths.unique(sorted=True).tolist():
        positions = (lengths == size).nonzero().flatten()
        compact = ids.index_select(1, positions).gather(
            -1, order.index_select(1, positions)[..., :size]
        )
        groups.append(UnionGroup(positions, compact))
    return groups


def partition_log_probs(logits, token_ids, chunk_size=128, output_dtype=torch.float32):
    """Full-vocabulary normalized selected probabilities plus the true tail.

    logits: [1, T, V]; token_ids: [1, T, S], unique IDs per position, S < V.
    No tail floor or selected-only normalization is applied. FP64 inputs retain
    FP64 reductions; other input dtypes use FP32 reductions. The default FP32
    output matches the training scorer; FP64 output is available for references.
    """
    if token_ids.shape[-1] >= logits.shape[-1]:
        raise ValueError("The partition requires a nonempty vocabulary complement")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    batch, length, vocab = logits.shape
    support = token_ids.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_ids = token_ids.reshape(-1, support)
    out = torch.empty(
        (flat_logits.shape[0], support + 1), dtype=output_dtype, device=logits.device
    )
    reduction_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    for start in range(0, flat_logits.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_logits.shape[0])
        values = flat_logits[start:stop].to(reduction_dtype)
        centered = values - values.amax(dim=-1, keepdim=True).detach()
        log_z = centered.logsumexp(dim=-1, keepdim=True)
        selected = centered.gather(dim=-1, index=flat_ids[start:stop]) - log_z
        excluded = centered.scatter(dim=-1, index=flat_ids[start:stop], value=-torch.inf)
        tail = excluded.logsumexp(dim=-1, keepdim=True) - log_z
        out[start:stop] = torch.cat((selected, tail), dim=-1).to(output_dtype)
    return out.reshape(batch, length, support + 1)


@torch.no_grad()
def cf_target_logs(full_log, cf_log, alpha=1.0):
    """Return log Q_F and Delta = log B - log Q_F on the shared partition."""
    full_log = full_log - full_log.logsumexp(-1, keepdim=True)
    cf_log = cf_log - cf_log.logsumexp(-1, keepdim=True)
    candidate = full_log + alpha * (full_log - cf_log)
    candidate = candidate - candidate.logsumexp(-1, keepdim=True)
    return full_log, candidate - full_log


@torch.no_grad()
def cf_mirror_coefficient(full_log, delta):
    """Maximize c in [0,1] subject to kappa(c) <= kappa(-1).

    kappa(c) = KL(Q_F || softmax(log Q_F + c * Delta)). The calculation uses
    FP64 and 32 bisection steps, returning the feasible endpoint. The 1e-12
    comparison tolerance is numerical, not a tunable KL budget.
    """
    full = full_log.detach().double().log_softmax(-1)
    direction = delta.detach().double()
    centered = direction - (full.exp() * direction).sum(-1, keepdim=True)

    def divergence(coefficient):
        return (full + coefficient.unsqueeze(-1) * centered).logsumexp(-1).clamp_min(0)

    one = torch.ones_like(full[..., 0])
    budget = divergence(-one)
    active = divergence(one) > budget + 1e-12
    low, high = torch.zeros_like(one), one.clone()
    if bool(active.any()):
        for _ in range(32):
            middle = (low + high) * 0.5
            feasible = divergence(middle) <= budget
            low = torch.where(feasible, middle, low)
            high = torch.where(feasible, high, middle)
    return torch.where(active, low, one)


def mcc_partition_loss(student_log, full_log, cf_log, alpha=1.5):
    """Return per-token reverse KL and the detached FP64 coefficient.

    All inputs use the same explicit tokens and tail cell. Gradients flow only
    through student_log. The target is constructed with FP64 coefficients, as
    in the training implementation.
    """
    full_log, delta = cf_target_logs(full_log, cf_log, alpha)
    with torch.no_grad():
        coefficient = cf_mirror_coefficient(full_log, delta)
        target_log = full_log.detach() + coefficient.unsqueeze(-1) * delta.detach()
        target_log = target_log - target_log.logsumexp(-1, keepdim=True)
    loss = (student_log.exp() * (student_log - target_log)).sum(-1)
    return loss, coefficient


def mcc_token_loss(student_logits, full_teacher_logits, cf_teacher_logits,
                   *, topk=64, alpha=1.5, chunk_size=128):
    """Compute MCC-OPD from three aligned response-position logits [1, T, V].

    Teacher views must score the SAME student-generated continuation. The full
    teacher uses all evidence; the counterfactual teacher uses ablated evidence.
    Both share the student's vocabulary and temperature convention. This helper
    materializes all three logit tensors; it is not the FSDP forwarding adapter.

    Returns FP32 per-token losses [1,T] (matching the training output) and FP64
    coefficients [1,T]. Apply a response mask and global token-mean reduction
    outside this function. For a memory-efficient scorer, call the partition
    API with separately computed teacher partition log-probabilities instead.
    """
    if student_logits.ndim != 3 or student_logits.shape[0] != 1:
        raise ValueError("Expected one response per call, with logits shaped [1, T, V]")
    if (full_teacher_logits.shape != student_logits.shape
            or cf_teacher_logits.shape != student_logits.shape):
        raise ValueError("Student and teacher logits must have matching response/vocabulary axes")
    if not 0 < topk < student_logits.shape[-1]:
        raise ValueError("Expected 0 < topk < vocabulary size")
    if student_logits.shape[1] == 0 or chunk_size <= 0:
        raise ValueError("Expected at least one response position and a positive chunk_size")
    with torch.no_grad():
        groups = build_union_groups(
            student_logits.detach().topk(topk, dim=-1).indices,
            full_teacher_logits.detach().topk(topk, dim=-1).indices,
        )
    losses = student_logits.new_zeros(student_logits.shape[:2], dtype=torch.float32)
    coefficients = student_logits.new_zeros(student_logits.shape[:2], dtype=torch.float64)
    for group in groups:
        # Chunk before index_select so temporary vocabulary tensors stay bounded.
        for start in range(0, group.positions.numel(), chunk_size):
            positions = group.positions[start:start + chunk_size]
            ids = group.token_ids[:, start:start + chunk_size]
            p = partition_log_probs(student_logits.index_select(1, positions), ids, chunk_size)
            with torch.no_grad():
                q_full = partition_log_probs(
                    full_teacher_logits.index_select(1, positions), ids, chunk_size
                )
                q_cf = partition_log_probs(
                    cf_teacher_logits.index_select(1, positions), ids, chunk_size
                )
            loss, coefficient = mcc_partition_loss(p, q_full, q_cf, alpha)
            losses = losses.index_copy(1, positions, loss.float())
            coefficients = coefficients.index_copy(1, positions, coefficient)
    return losses, coefficients
