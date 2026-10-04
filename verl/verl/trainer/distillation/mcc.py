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
# Modified for MCC-OPD: actor-local FSDP teacher and response-only distillation.

"""MCC-OPD forwarding adapter for the synchronous verl V1 FSDP trainer.

The frozen teacher scores the same student continuation with full and ablated
memory. Only response logits are materialized; reduction remains verl's global
valid-token mean. The tensor target is shared with the standalone core.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import torch

from verl.trainer.distillation.mcc_core import (
    build_union_groups, partition_log_probs, mcc_partition_loss,
)
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import no_padding_2_padding

LOSS_MODE = "mcc_opd"


def is_mcc_distillation(config):
    return config is not None and config.enabled and config.distillation_loss.loss_mode == LOSS_MODE


def response_prediction_positions(data) -> torch.Tensor:
    """复用原 scorer 的因果位移；首个 response token 由 prompt 的最后位置预测。"""
    # ponytail: 当前实测配方为 microbatch 1、SP1、无 packing；不扩展到其他布局。
    if data["input_ids"].shape[0] != 1:
        raise ValueError("response_only currently requires microbatch size 1")
    packed = torch.arange(data["input_ids"].values().numel(), device=data["input_ids"].device)
    mask = data["response_mask"].bool()
    if mask.is_nested:
        mask = mask.to_padded_tensor(False)
    positions = no_padding_2_padding(packed, data)[mask]
    # 全 mask 的 microbatch 仍计算一个零权重位置，以保留反向图与 FSDP 同步。
    return positions if positions.numel() else packed[:1]

def _score_union_groups(logits, groups, chunk_size):
    """仅复制有限位置chunk的词表logits；Student调用保留index_select的反向图。"""
    result = []
    for group in groups:
        chunks = []
        for start in range(0, group.positions.numel(), chunk_size):
            positions = group.positions[start : start + chunk_size]
            chunks.append(
                partition_log_probs(
                    logits.index_select(1, positions),
                    group.token_ids[:, start : start + chunk_size],
                    chunk_size,
                    output_dtype=torch.float32,
                )
            )
        result.append(torch.cat(chunks, dim=1))
    return result

def counterfactual_microbatch(data, prompt_ids=None):
    """只替换 prompt；直接复用未重分词的 Student response，包括 mask=0 的前缀 token。"""
    if data["input_ids"].shape[0] != 1 or not data["responses"].is_nested:
        raise ValueError("MCC counterfactual scoring requires a V1 nested microbatch of size 1")
    if prompt_ids is None:
        prompt_ids = data["extra_info"][0]["vad_cf_input_ids"]
    prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=data["input_ids"].device)
    if prompt.ndim != 1 or prompt.numel() == 0:
        raise ValueError("extra_info.vad_cf_input_ids must contain a nonempty tokenized prompt")
    ids = torch.cat((prompt, data["responses"][0]))
    result = data.clone(recurse=False)
    result["prompts"] = torch.nested.as_nested_tensor([prompt], layout=torch.jagged)
    result["input_ids"] = torch.nested.as_nested_tensor([ids], layout=torch.jagged)
    result["position_ids"] = torch.nested.as_nested_tensor(
        [torch.arange(ids.numel(), device=ids.device)], layout=torch.jagged
    )
    return result

class FrozenFSDPTeacher:
    """Actor-local frozen Qwen3 teacher, with no optimizer or checkpoint."""

    def __init__(self, actor_config, teacher_config, chunk_size, response_only=False):
        from verl.workers.config import HFModelConfig
        from verl.workers.engine import EngineRegistry

        engine_config = actor_config.engine
        if engine_config.strategy not in ("fsdp", "fsdp2"):
            raise ValueError(f"{LOSS_MODE} currently requires FSDP")
        # ponytail: 首版仅 SP=1；扩展 SP 时须同步切分 Teacher 的指定 IDs。
        if engine_config.ulysses_sequence_parallel_size != 1 or engine_config.pad_to_length:
            raise ValueError(f"{LOSS_MODE} requires SP=1 and pad_to_length=False")
        if actor_config.model_config.use_fused_kernels:
            raise ValueError(f"{LOSS_MODE} requires eager logits (use_fused_kernels=False)")

        student_model = actor_config.model_config
        if response_only and (
            engine_config.use_remove_padding
            or engine_config.use_dynamic_bsz
            or engine_config.micro_batch_size_per_gpu != 1
            or student_model.hf_config.model_type != "qwen3"
        ):
            raise ValueError("response_only requires Qwen3, no remove_padding, static microbatch size 1")
        teacher_model = HFModelConfig(
            path=teacher_config.model_path,
            trust_remote_code=student_model.trust_remote_code,
            use_shm=student_model.use_shm,
            use_remove_padding=student_model.use_remove_padding,
            enable_gradient_checkpointing=False,
            override_config={
                "attn_implementation": student_model.hf_config._attn_implementation,
            },
        )
        if (
            teacher_model.hf_config.vocab_size != student_model.hf_config.vocab_size
            or teacher_model.tokenizer.get_vocab() != student_model.tokenizer.get_vocab()
        ):
            raise ValueError("Student and Teacher must share the same vocabulary and token IDs")
        if hasattr(teacher_model.hf_config, "vision_config") or hasattr(student_model.hf_config, "vision_config"):
            raise ValueError(f"{LOSS_MODE} currently supports text models only")
        if response_only and teacher_model.hf_config.model_type != "qwen3":
            raise ValueError("response_only currently requires a Qwen3 Teacher")

        self.keep_on_gpu_during_update = teacher_config.keep_on_gpu_during_update
        teacher_engine_config = replace(
            deepcopy(engine_config),
            forward_only=True,
            model_dtype="bf16",
            use_fused_kernels=False,
            forward_only_cpu_offload=not self.keep_on_gpu_during_update,
            param_offload=True,
            optimizer_offload=False,
            offload_policy=False,
        )
        self.engine = EngineRegistry.new(
            model_type="language_model",
            backend=teacher_engine_config.strategy,
            model_config=teacher_model,
            engine_config=teacher_engine_config,
            optimizer_config=None,
            checkpoint_config=actor_config.checkpoint,
        )
        self.engine.initialize()
        self.engine.module.requires_grad_(False)
        self.chunk_size = chunk_size

    @contextmanager
    def update_context(self):
        if not self.keep_on_gpu_during_update:
            yield
            return
        from verl.utils.device import get_device_name

        with torch.profiler.record_function("opd/teacher_load"):
            self.engine.to(get_device_name(), optimizer=False, grad=False)
        try:
            yield
        finally:
            with torch.profiler.record_function("opd/teacher_offload"):
                self.engine.to("cpu", optimizer=False, grad=False)

    @torch.no_grad()
    def score_union_views(self, data, student_ids, cf_prompt_ids, teacher_topk):
        """一次full前向同时选择Teacher候选并评分；各CF复用同一真实union。"""
        def full_scores(logits):
            teacher_ids = logits.topk(teacher_topk, dim=-1).indices
            groups = build_union_groups(student_ids, teacher_ids)
            return groups, _score_union_groups(logits, groups, self.chunk_size)

        # MCC always scores two views on every FSDP rank, including padding rows.
        count = 1 + len(cf_prompt_ids)
        groups, full = self._forward_scores(data, full_scores)
        views = [full]
        for index in range(1, count):
            views.append(
                self._forward_scores(
                    counterfactual_microbatch(data, cf_prompt_ids[index - 1]),
                    lambda logits: _score_union_groups(logits, groups, self.chunk_size),
                )
            )
        return groups, views

    @torch.no_grad()
    def _forward_scores(self, data, score_fn):
        # 两端复用同一个 microbatch、packing 和位置编码；只对 Teacher 关闭梯度。
        with self.engine.eval_mode(disable_auto_offload=self.keep_on_gpu_during_update):
            model_inputs, output_args = self.engine.prepare_model_inputs(data)
            response_only = tu.get_non_tensor_data(data, "mcc_response_only", default=False)
            if response_only:
                model_inputs["logits_to_keep"] = response_prediction_positions(data)
            with torch.profiler.record_function("opd/teacher_forward"):
                logits = self.engine.module(**model_inputs, use_cache=False).logits
            if response_only:
                temperature = output_args["temperature"][:, None]
            elif self.engine.engine_config.use_remove_padding:
                temperature = output_args["temperature_rmpad"]
            else:
                lengths = data["input_ids"].offsets().diff()
                logits = torch.cat([row[:length] for row, length in zip(logits, lengths, strict=True)]).unsqueeze(0)
                temperature = torch.repeat_interleave(output_args["temperature"], lengths)
            logits = logits / temperature.clamp(min=1e-8).unsqueeze(-1).to(logits.dtype)
            with torch.profiler.record_function("opd/teacher_logprobs"):
                return score_fn(logits)


def mcc_token_outputs(student_logits, data, config, teacher):
    """Score the student/full-teacher top-K union and exact complement tail."""
    with torch.no_grad():
        student_ids = student_logits.detach().topk(config.topk, dim=-1).indices
    groups, views = teacher.score_union_views(
        data, student_ids, [data["extra_info"][0]["vad_cf_input_ids"]], config.topk,
    )
    student = _score_union_groups(student_logits, groups, config.chunked_topk_chunk_size)
    result = {
        key: student_logits.new_zeros(student_logits.shape[:2], dtype=torch.float32)
        for key in ("distillation_losses", "student_mass", "teacher_mass")
    }
    for group, p, full, cf in zip(groups, student, *views, strict=True):
        losses, _ = mcc_partition_loss(p, full, cf, alpha=config.alpha)
        values = {
            "distillation_losses": losses.float(),
            "student_mass": p[..., :-1].exp().sum(-1),
            "teacher_mass": full[..., :-1].exp().sum(-1),
        }
        for key, value in values.items():
            result[key] = result[key].index_copy(1, group.positions, value)
    return result
