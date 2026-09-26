# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
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

import logging as py_logging
import math
import os
import re
import sys
import textwrap
import warnings
import unicodedata
from collections.abc import Iterator
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import torch
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


def _text_debug_stats(text: str) -> dict[str, int]:
    stats = {
        "length": len(text),
        "cjk": 0,
        "latin": 0,
        "digit": 0,
        "thai": 0,
        "arabic": 0,
        "symbol": 0,
    }
    for char in text:
        code = ord(char)
        if "\u4e00" <= char <= "\u9fff":
            stats["cjk"] += 1
        elif "a" <= char.lower() <= "z":
            stats["latin"] += 1
        elif char.isdigit():
            stats["digit"] += 1
        elif 0x0E00 <= code <= 0x0E7F:
            stats["thai"] += 1
        elif 0x0600 <= code <= 0x06FF:
            stats["arabic"] += 1
        elif unicodedata.category(char).startswith("S"):
            stats["symbol"] += 1

    return stats


def _token_debug_snapshot(tokenizer: "PreTrainedTokenizer", token_ids: list[int], max_tokens: int = 24) -> str:
    head_ids = token_ids[:max_tokens]
    head_tokens = tokenizer.convert_ids_to_tokens(head_ids)
    paired = [f"{tid}:{repr(tok)}" for tid, tok in zip(head_ids, head_tokens)]
    if len(token_ids) > max_tokens:
        paired.append("…")

    return " | ".join(paired)


class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""Inherit PPOTrainer."""

    @property
    def tokenizer(self):  # type: ignore[override]
        return getattr(self, "processing_class", None)

    @tokenizer.setter
    def tokenizer(self, value):  # type: ignore[override]
        self.processing_class = value

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[list["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding",
        reward_callback: Optional[Callable[..., Sequence[float]]] = None,
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")

        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        # Add deepspeed config
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.reward_callback = reward_callback
        self.current_device = get_current_device()  # patch for deepspeed training

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )
        self._log_generation_diagnostics()

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        # Transformers>=4.44 推荐通过 processing_class 访问分词器，避免重复 Warning
        self.processing_class = tokenizer
        if self.args.max_steps > 0:
            logger.info_rank0("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def _log_generation_diagnostics(self) -> None:
        eos_ids = self.generation_config.eos_token_id
        if not isinstance(eos_ids, list):
            eos_ids = [eos_ids]

        logger.info_rank0(
            "PPO generation config: do_sample=%s temperature=%s top_p=%s top_k=%s repetition_penalty=%s max_new_tokens=%s",
            self.generation_config.do_sample,
            self.generation_config.temperature,
            self.generation_config.top_p,
            self.generation_config.top_k,
            self.generation_config.repetition_penalty,
            self.generation_config.max_new_tokens,
        )
        logger.debug_rank0(
            "PPO tokenizer ids: pad=%s bos=%s eos=%s additional_special_tokens=%s",
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.additional_special_tokens_ids,
        )

        if not self.generation_config.do_sample and any(
            value is not None
            for value in [
                self.generation_config.temperature,
                self.generation_config.top_p,
                self.generation_config.top_k,
            ]
        ):
            logger.warning_rank0(
                "PPO rollout 当前 do_sample=False，temperature/top_p/top_k 将被忽略。若想增加多样性，请显式设置 do_sample=True。"
            )

        if self.tokenizer.eos_token_id not in eos_ids:
            logger.warning_rank0(
                "tokenizer.eos_token_id=%s 不在 generation eos_token_id=%s 中，可能导致生成过长或无法按预期停止。",
                self.tokenizer.eos_token_id,
                eos_ids,
            )

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""Implement training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer."""
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running training *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info_rank0(
            f"  Total train batch size (w. parallel, buffer, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Num optimization epochs per batch = {self.finetuning_args.ppo_epochs:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            batch, dataiter = self._prepare_full_batch(batch, dataiter)

            # Get inputs
            self.model.eval()
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards, metas = [], [], [], []
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch = {
                    "input_ids": batch["input_ids"][idx : idx + self.config.mini_batch_size],
                    "attention_mask": batch["attention_mask"][idx : idx + self.config.mini_batch_size],
                }
                if "metas" in batch:
                    mini_batch["metas"] = batch["metas"][idx : idx + self.config.mini_batch_size]
                mini_batch_queries, mini_batch_responses, mini_batch_metas = self.get_inputs(mini_batch)
                mini_batch_rewards = self.get_rewards(
                    mini_batch_queries, mini_batch_responses, metas=mini_batch_metas
                )
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)
                metas.extend(mini_batch_metas)

            # Run PPO step
            self.model.train()
            stats = self.step(queries, responses, rewards)
            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning_rank0("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    def _prepare_full_batch(
        self, batch: dict[str, Any], dataiter: Iterator[dict[str, Any]]
    ) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
        target_size = self.config.batch_size
        input_ids = batch.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            return batch, dataiter

        current_size = input_ids.size(0)
        if current_size == target_size:
            return batch, dataiter

        merged_batch = self._clone_batch(batch)
        while current_size < target_size:
            try:
                next_chunk = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                next_chunk = next(dataiter)

            merged_batch = self._concat_batch(merged_batch, next_chunk)
            current_size = merged_batch["input_ids"].size(0)

        if current_size > target_size:
            merged_batch = self._trim_batch(merged_batch, target_size)

        return merged_batch, dataiter

    @staticmethod
    def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
        cloned_batch: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cloned_batch[key] = value.clone()
            elif isinstance(value, list):
                cloned_batch[key] = value.copy()
            elif isinstance(value, tuple):
                cloned_batch[key] = list(value)
            else:
                cloned_batch[key] = value
        return cloned_batch

    @staticmethod
    def _concat_batch(base: dict[str, Any], addition: dict[str, Any]) -> dict[str, Any]:
        for key, value in addition.items():
            if isinstance(value, torch.Tensor):
                if key in base and isinstance(base[key], torch.Tensor):
                    base[key] = torch.cat((base[key], value), dim=0)
                else:
                    base[key] = value.clone()
            elif isinstance(value, list):
                existing = base.get(key, [])
                if isinstance(existing, list):
                    base[key] = existing + value
                else:
                    base[key] = value.copy()
            elif isinstance(value, tuple):
                existing = base.get(key, [])
                if isinstance(existing, list):
                    base[key] = existing + list(value)
                else:
                    base[key] = list(value)
            else:
                base.setdefault(key, value)
        return base

    @staticmethod
    def _trim_batch(batch: dict[str, Any], target_size: int) -> dict[str, Any]:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value[:target_size]
            elif isinstance(value, list):
                batch[key] = value[:target_size]
        return batch

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(
        self, batch: dict[str, Any]
    ) -> tuple[list["torch.Tensor"], list["torch.Tensor"], list[Optional[Any]]]:
        r"""Generate model's responses given queries."""
        metas = batch.get("metas")
        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v[:, start_index:]

        logger.debug(
            "Preparing PPO generation batch: input_ids_shape=%s, attention_mask_shape=%s, meta_count=%s",
            tuple(batch["input_ids"].shape) if "input_ids" in batch else None,
            tuple(batch["attention_mask"].shape) if "attention_mask" in batch else None,
            len(metas) if isinstance(metas, list) else None,
        )

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)

            generate_output: torch.Tensor = unwrapped_model.generate(
                generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
            )
            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_indexes) == 0:  # allow empty response
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:  # include eos token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right

        if isinstance(metas, list):
            meta_list: list[Optional[Any]] = metas
        else:
            meta_list = [None for _ in queries]

        logger.debug("Decoded PPO responses: query_count=%d, response_count=%d", len(queries), len(responses))

        if logger.isEnabledFor(py_logging.INFO):
            for idx, (query_ids, response_ids) in enumerate(zip(queries, responses)):
                prompt_text = self.tokenizer.decode(query_ids, skip_special_tokens=True)
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                logger.info_rank0(
                    "PPO 样本 %d\n[Prompt]\n%s\n[Response]\n%s",
                    idx,
                    prompt_text.strip() or "<empty>",
                    response_text.strip() or "<empty>",
                )

        return queries, responses, meta_list

    @torch.no_grad()
    def get_rewards(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
        metas: Optional[list[Optional[Any]]] = None,
    ) -> list["torch.Tensor"]:
        r"""Compute scores using given reward model.

        Both inputs and outputs are put on CPU.
        """
        if self.reward_callback is not None:
            meta_list = metas or [None for _ in queries]
            formatted_metas = [m if isinstance(m, dict) else {} for m in meta_list]
            sequences = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
            if self.tokenizer is not None:
                prompt_texts = self.tokenizer.batch_decode([q.tolist() for q in queries], skip_special_tokens=True)
                response_texts = self.tokenizer.batch_decode(
                    [r.tolist() for r in responses], skip_special_tokens=True
                )
                for meta, prompt_text, response_text, response_ids in zip(
                    formatted_metas, prompt_texts, response_texts, responses
                ):
                    if isinstance(meta, dict):
                        meta.setdefault("prompt", prompt_text)
                        meta.setdefault("response", response_text)
                        meta.setdefault("raw_response", response_text)
                        if "trajectory_id" not in meta:
                            match = re.search(r"\[TRAJ\]\s*([^\s]+)", prompt_text)
                            if match:
                                meta["trajectory_id"] = match.group(1)
                    trajectory_id = meta.get("trajectory_id") if isinstance(meta, dict) else None
                    summary_prompt = textwrap.shorten(prompt_text.strip().replace("\n", " "), width=400, placeholder="…")
                    summary_response = textwrap.shorten(
                        response_text.strip().replace("\n", " "), width=400, placeholder="…"
                    )
                    logger.info_rank0(
                        "[PPO] 轨迹=%s\n  问题: %s\n  回复: %s",
                        trajectory_id or "<unknown>",
                        summary_prompt or "<empty>",
                        summary_response or "<empty>",
                    )
                    response_stats = _text_debug_stats(response_text)
                    logger.debug_rank0(
                        "[PPO] response_stats 轨迹=%s len=%d cjk=%d latin=%d digit=%d thai=%d arabic=%d symbol=%d",
                        trajectory_id or "<unknown>",
                        response_stats["length"],
                        response_stats["cjk"],
                        response_stats["latin"],
                        response_stats["digit"],
                        response_stats["thai"],
                        response_stats["arabic"],
                        response_stats["symbol"],
                    )
                    if response_stats["length"] > 0 and response_stats["cjk"] == 0 and response_stats["latin"] < 8:
                        logger.warning_rank0(
                            "[PPO] 轨迹=%s 的回复几乎不含中英文字符，可能是采样配置、模板不匹配或底模/词表未对齐。",
                            trajectory_id or "<unknown>",
                        )
                        logger.debug_rank0(
                            "[PPO] 轨迹=%s response_token_snapshot: %s",
                            trajectory_id or "<unknown>",
                            _token_debug_snapshot(self.tokenizer, response_ids.tolist()),
                        )
            try:
                rewards = self.reward_callback(
                    sequences=sequences,
                    metas=formatted_metas,
                    logprobs=None,
                )
            except TypeError:
                rewards = self.reward_callback(sequences, formatted_metas)  # type: ignore[misc]

            tensors: list[torch.Tensor] = []
            for reward in rewards:
                if isinstance(reward, torch.Tensor):
                    tensors.append(reward.detach().to(torch.float32))
                else:
                    tensors.append(torch.tensor(float(reward), dtype=torch.float32))
            return tensors

        if self.finetuning_args.reward_model_type == "api":
            token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
            messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
            return get_rewards_from_server(self.reward_model, messages)

        batch: dict[str, torch.Tensor] = self.prepare_model_inputs(queries, responses)
        unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)

        if self.finetuning_args.reward_model_type in ["lora", "oft"]:
            replace_model(unwrapped_model, target="reward")
            reward_model = self.model
        else:
            reward_model = self.reward_model

        with unwrap_model_for_generation(reward_model, self.accelerator), self.amp_context:  # support bf16
            values: torch.Tensor = reward_model(**batch, return_dict=True, use_cache=False)[-1]

        if self.finetuning_args.reward_model_type in ["lora", "oft"]:
            replace_model(unwrapped_model, target="default")

        rewards = values.gather(dim=-1, index=(batch["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return rewards.float().detach()  # use fp32 type

    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""Calculate model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with self.amp_context:  # support bf16
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""Save model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())
