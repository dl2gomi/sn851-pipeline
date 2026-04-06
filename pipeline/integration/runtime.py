from __future__ import annotations

import asyncio
import copy
import json
import logging
import random
import threading
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List
from collections.abc import Generator

from ..config import TrainingConfig
from ..schemas import Task, Trajectory


class SharedModelRuntime:
    """Single-process runtime with shared model state for inference + RL updates."""

    def __init__(
        self,
        model_dir: Path,
        training_defaults: TrainingConfig,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        serve_host: str = "127.0.0.1",
        serve_port: int = 8000,
        serve_api_key: str = "local-dev",
    ):
        self.model_dir = Path(model_dir)
        self.training_defaults = training_defaults
        self.max_new_tokens = max_new_tokens
        self.temperature = float(self.training_defaults.temperature)
        self.top_p = top_p
        self.serve_host = serve_host
        self.serve_port = serve_port
        self.serve_api_key = serve_api_key

        self._tokenizer: Any = None
        self._model: Any = None
        self._reference_model: Any = None
        self._value_head: Any = None
        self._optimizer: Any = None
        self._torch: Any = None
        self._weights_updated_at: float | None = None
        self._model_lock = threading.RLock()

        self._server: Any = None
        self._server_thread: threading.Thread | None = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "SharedModelRuntime requires torch and transformers. "
                "Install them before non-dry training runs."
            ) from exc

        if not self.model_dir.exists():
            raise ValueError(f"Model directory does not exist: {self.model_dir}")

        self._torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        self._model.train()

        # Load training state if exists
        state_path = self.model_dir / 'training_state.pt'
        if state_path.exists():
            state = torch.load(state_path, map_location=device)
            # Optimizer and scheduler will be loaded after they are created

        # Reference model for KL regularization anchor (frozen snapshot).
        self._reference_model = copy.deepcopy(self._model).eval()
        for p in self._reference_model.parameters():
            p.requires_grad = False

        hidden = int(getattr(self._model.config, "hidden_size", 0) or 0)
        if self.training_defaults.use_value_head:
            if hidden <= 0:
                raise ValueError("use_value_head requires model.config.hidden_size")
            self._value_head = torch.nn.Linear(hidden, 1, dtype=torch.float32).to(device)
        else:
            self._value_head = None

        opt_params = list(self._model.parameters())
        if self._value_head is not None:
            opt_params += list(self._value_head.parameters())
        self._optimizer = torch.optim.AdamW(opt_params, lr=float(self.training_defaults.lr))

        if self.training_defaults.lr_scheduler == "cosine":
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, T_max=self.training_defaults.max_steps
            )
        else:
            self._scheduler = None

        self._kl_coef = float(self.training_defaults.kl_coef)
        self._entropy_coef = float(self.training_defaults.entropy_coef)

        if 'state' in locals() and state:
            self._optimizer.load_state_dict(state['optimizer'])
            if self._scheduler and 'scheduler' in state and state['scheduler']:
                self._scheduler.load_state_dict(state['scheduler'])
            self._kl_coef = state.get('kl_coef', self._kl_coef)
            self._entropy_coef = state.get('entropy_coef', self._entropy_coef)
            self.temperature = state.get('temperature', self.temperature)
            self._weights_updated_at = state.get('weights_updated_at', self._weights_updated_at)

        self._weights_updated_at = time.time()

    def generate(self, task: Task) -> str:
        return self.generate_from_prompt(task.prompt or "")

    def rollout_forward(self, task: Task) -> tuple[str, float, float]:
        """Sample completion + off-policy stats for PPO (under model lock)."""
        self._lazy_init()
        prompt = task.prompt or ""
        with self._model_lock:
            text = self._generate_unlocked(prompt)
            fake = Trajectory(task=task, response=text, raw_score=0.0, reward=0.0)
            item = self._encode_trajectory(fake)
            with self._torch.no_grad():
                lp_t = self._response_log_prob_sum(
                    item["input_ids"], item["labels"], use_reference=False
                )
                lp = float(lp_t.item())
                if self._value_head is not None:
                    v = float(self._value_at_prompt(prompt, requires_grad=False).item())
                else:
                    v = 0.0
        return text, lp, v

    def generate_from_prompt(self, prompt: str) -> str:
        self._lazy_init()
        with self._model_lock:
            return self._generate_unlocked(prompt)

    def _build_prompt_from_messages(self, messages: Any) -> str:
        """Turn OpenAI-style chat messages into a single model prompt (chat template when available)."""
        if not isinstance(messages, list) or not messages:
            return ""
        normalized: list[dict[str, str]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "user"))
            content = m.get("content")
            if content is None:
                continue
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                content = "".join(parts) if parts else str(content)
            else:
                content = str(content)
            normalized.append({"role": role, "content": content})
        if not normalized:
            return ""
        tok = self._tokenizer
        if getattr(tok, "chat_template", None) is not None and hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        lines = [f"{m['role'].upper()}: {m['content']}" for m in normalized]
        return "\n".join(lines)

    @staticmethod
    def _normalize_stop(stop: Any) -> list[str] | None:
        if stop is None:
            return None
        if isinstance(stop, str):
            return [stop] if stop else None
        try:
            out = [str(s) for s in stop if s]
        except TypeError:
            return None
        return out or None

    @staticmethod
    def _apply_stop_strings(text: str, stops: list[str] | None) -> tuple[str, bool]:
        if not stops or not text:
            return text, False
        earliest = len(text)
        for s in stops:
            if not s:
                continue
            i = text.find(s)
            if i != -1 and i < earliest:
                earliest = i
        if earliest >= len(text):
            return text, False
        return text[:earliest], True

    def _generate_unlocked(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> str:
        text, _, _, _ = self._generate_unlocked_with_meta(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            seed=seed,
        )
        return text

    def _generate_unlocked_with_meta(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> tuple[str, int, int, str]:
        """Returns (text, prompt_tokens, completion_tokens, finish_reason)."""
        torch = self._torch
        if seed is not None:
            torch.manual_seed(int(seed))
        mnt = int(max_new_tokens) if max_new_tokens is not None else self.max_new_tokens
        mnt = max(1, min(mnt, 1_000_000))
        temp = float(self.temperature if temperature is None else temperature)
        tp = float(self.top_p if top_p is None else top_p)
        do_sample = temp is not None and temp > 1e-6

        model_inputs = self._tokenizer(prompt, return_tensors="pt")
        model_inputs = {k: v.to(self._model.device) for k, v in model_inputs.items()}
        prompt_len = int(model_inputs["input_ids"].shape[1])

        gen_kw: dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": mnt,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kw["temperature"] = max(temp, 1e-6)
            gen_kw["top_p"] = tp

        with torch.no_grad():
            out = self._model.generate(**gen_kw)
        new_tokens = int(out.shape[1]) - prompt_len
        raw_ids = out[0, prompt_len:]
        text = self._tokenizer.decode(raw_ids, skip_special_tokens=True)
        text, hit_stop = self._apply_stop_strings(text, stop)

        comp_tokens = len(self._tokenizer.encode(text, add_special_tokens=False)) if text else 0
        eos_id = self._tokenizer.eos_token_id
        natural_eos = (
            eos_id is not None
            and raw_ids.numel() > 0
            and int(raw_ids[-1].item()) == int(eos_id)
        )

        if hit_stop or natural_eos:
            finish = "stop"
        elif new_tokens >= mnt:
            finish = "length"
        else:
            finish = "stop"

        return text, prompt_len, comp_tokens, finish

    def _stream_chat_unlocked(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> Any:
        """Yields (delta_text, is_final, usage_dict|None, finish_reason|None)."""
        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            text, pt, ct, fr = self._generate_unlocked_with_meta(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                seed=seed,
            )
            if text:
                yield text, False, None, None
            yield "", True, {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}, fr
            return

        torch = self._torch
        if seed is not None:
            torch.manual_seed(int(seed))
        mnt = int(max_new_tokens) if max_new_tokens is not None else self.max_new_tokens
        mnt = max(1, min(mnt, 1_000_000))
        temp = float(self.temperature if temperature is None else temperature)
        tp = float(self.top_p if top_p is None else top_p)
        do_sample = temp is not None and temp > 1e-6

        model_inputs = self._tokenizer(prompt, return_tensors="pt")
        model_inputs = {k: v.to(self._model.device) for k, v in model_inputs.items()}
        prompt_len = int(model_inputs["input_ids"].shape[1])

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kw: dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": mnt,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }
        if do_sample:
            gen_kw["temperature"] = max(temp, 1e-6)
            gen_kw["top_p"] = tp

        def _worker() -> None:
            with torch.no_grad():
                self._model.generate(**gen_kw)

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        emitted = 0
        full = ""
        for new_text in streamer:
            full += new_text
            display, hit = self._apply_stop_strings(full, stop)
            if len(display) > emitted:
                yield display[emitted:], False, None, None
                emitted = len(display)
            if hit:
                pt = prompt_len
                ct = len(self._tokenizer.encode(display, add_special_tokens=False))
                yield "", True, {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}, "stop"
                worker.join(timeout=600)
                return
        worker.join(timeout=600)
        display, hit_stop = self._apply_stop_strings(full, stop)
        if len(display) > emitted:
            yield display[emitted:], False, None, None
        ct = len(self._tokenizer.encode(display, add_special_tokens=False))
        pt = prompt_len
        yield "", True, {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}, "stop"

    @property
    def model_name(self) -> str:
        return self.model_dir.name

    @property
    def base_url(self) -> str:
        return f"http://{self.serve_host}:{self.serve_port}/v1"

    @property
    def weights_updated_at(self) -> float | None:
        return self._weights_updated_at

    def save_hf_checkpoint(
        self,
        output_dir: Path,
        *,
        safe_serialization: bool = True,
        max_shard_size: str = "10GB",
    ) -> dict[str, Any]:
        """Write model + tokenizer to a directory (under model lock). Returns manifest fields."""
        self._lazy_init()
        output_dir = Path(output_dir)
        meta: dict[str, Any] = {
            "source_model_dir": str(self.model_dir.resolve()),
            "weights_updated_at": self._weights_updated_at,
            "weights_updated_at_iso": None,
        }
        if self._weights_updated_at is not None:
            meta["weights_updated_at_iso"] = datetime.fromtimestamp(
                self._weights_updated_at, tz=timezone.utc
            ).isoformat()

        with self._model_lock:
            was_training = self._model.training
            self._model.eval()
            try:
                self._model.save_pretrained(
                    output_dir,
                    safe_serialization=safe_serialization,
                    max_shard_size=max_shard_size,
                )
                self._tokenizer.save_pretrained(output_dir)
            finally:
                self._model.train(was_training)
        # Save training state
        torch.save({
            'optimizer': self._optimizer.state_dict(),
            'scheduler': self._scheduler.state_dict() if self._scheduler else None,
            'kl_coef': self._kl_coef,
            'entropy_coef': self._entropy_coef,
            'temperature': self.temperature,
            'weights_updated_at': self._weights_updated_at
        }, output_dir / 'training_state.pt')
        return meta

    def ensure_server_started(self) -> None:
        if self._server_thread and self._server_thread.is_alive():
            return
        self._lazy_init()
        self._start_server()

    def stop_server(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=10)
        self._server = None
        self._server_thread = None

    def train_step(self, batch: List[Trajectory], training: TrainingConfig) -> float:
        self._lazy_init()
        torch = self._torch
        if not batch:
            return 0.0

        algo = (training.algorithm or "ppo").lower()
        if algo not in ("ppo", "a2c"):
            warnings.warn(
                f"training.algorithm={training.algorithm!r} is not fully supported; using PPO clipped objective.",
                RuntimeWarning,
                stacklevel=2,
            )
            algo = "ppo"

        device = self._model.device
        rewards = torch.tensor([float(t.reward) for t in batch], device=device, dtype=torch.float32)

        # Reward normalization for stability
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-8
        normalized_rewards = (rewards - reward_mean) / reward_std

        # Compute GAE advantages
        gamma = float(training.gamma)
        gae_lambda = float(training.gae_lambda)
        advantages_list = []
        returns_list = []
        for traj in batch:
            adv = 0.0
            for t in reversed(range(len(traj.rewards))):
                next_v = 0.0 if t == len(traj.rewards) - 1 or traj.dones[t] else traj.values[t + 1]
                delta = traj.rewards[t] + gamma * next_v - traj.values[t]
                adv = delta + gamma * gae_lambda * adv
            advantages_list.append(adv)
            # Compute return: advantage + baseline value
            ret = adv + traj.values[0]
            returns_list.append(ret)
        advantages = torch.tensor(advantages_list, device=device, dtype=torch.float32)
        returns = torch.tensor(returns_list, device=device, dtype=torch.float32)

        # Normalize advantages
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = advantages / (adv_std + 1e-8)

        encoded = [self._encode_trajectory(t) for t in batch]
        old_logps: list[Any] = []
        with torch.no_grad():
            for t, item in zip(batch, encoded):
                if abs(float(t.rollout_logprob_sum)) > 1e-12:
                    old_logps.append(
                        torch.tensor(float(t.rollout_logprob_sum), device=device, dtype=torch.float32)
                    )
                else:
                    old_logps.append(
                        self._response_log_prob_sum(
                            item["input_ids"], item["labels"], use_reference=False
                        ).float()
                    )

        with self._model_lock:
            grad_accum = max(1, int(training.gradient_accumulation_steps))
            ppo_epochs = max(1, int(training.ppo_epochs))
            kl_coef = self._kl_coef
            entropy_coef = self._entropy_coef
            max_grad_norm = float(training.max_grad_norm)
            clip_eps = float(training.clip_range)
            target_kl = float(training.target_kl)
            value_loss_coef = float(training.value_loss_coef)

            total_loss = 0.0
            updates = 0
            self._optimizer.zero_grad(set_to_none=True)

            param_list = list(self._model.parameters())
            if self._value_head is not None:
                param_list += list(self._value_head.parameters())

            for _epoch in range(ppo_epochs):
                epoch_kls: list[float] = []
                order = list(range(len(encoded)))
                random.shuffle(order)
                mini_batch_size = max(1, int(training.mini_batch_size))
                for mini_start in range(0, len(order), mini_batch_size):
                    mini_indices = order[mini_start:mini_start + mini_batch_size]
                    mini_loss = 0.0
                    mini_approx_kls = []
                    for i in mini_indices:
                        item = encoded[i]
                        new_lp = self._response_log_prob_sum(
                            item["input_ids"], item["labels"], use_reference=False
                        )
                        old_lp = old_logps[i].to(device=new_lp.device, dtype=new_lp.dtype)
                        ratio = torch.exp(new_lp - old_lp).squeeze()
                        adv = advantages[i].to(dtype=ratio.dtype)
                        unclipped = ratio * adv
                        if algo == "ppo":
                            clipped_r = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                            policy_loss = -torch.min(unclipped, clipped_r)
                        else:  # a2c
                            policy_loss = -unclipped

                        nll_cur = self._response_nll(item["input_ids"], item["labels"], use_reference=False)
                        with torch.no_grad():
                            nll_ref = self._response_nll(item["input_ids"], item["labels"], use_reference=True)
                        kl_loss = (nll_cur - nll_ref) ** 2

                        entropy = self._response_entropy(item["input_ids"], item["labels"], use_reference=False)

                        loss = policy_loss + kl_coef * kl_loss - entropy_coef * entropy
                        if use_v:
                            v_pred = self._value_at_prompt(batch[i].task.prompt or "", requires_grad=True)
                            if v_pred.dim() > 0:
                                v_pred = v_pred.squeeze()
                            if algo == "ppo":
                                old_v = batch[i].rollout_value
                                v_clipped = torch.clamp(v_pred, old_v - clip_eps, old_v + clip_eps)
                                value_loss = torch.max((v_pred - returns[i]) ** 2, (v_clipped - returns[i]) ** 2)
                            else:  # a2c
                                value_loss = (v_pred - returns[i]) ** 2
                            loss = loss + value_loss_coef * value_loss

                        approx_kl = float((old_lp - new_lp.detach()).reshape(-1)[0].item())
                        mini_approx_kls.append(approx_kl)
                        mini_loss += loss

                    mini_loss /= len(mini_indices)
                    (mini_loss / grad_accum).backward()
                    total_loss += float(mini_loss.detach().reshape(-1)[0].item()) * len(mini_indices)
                    updates += len(mini_indices)
                    epoch_kls.extend(mini_approx_kls)

                    if ((mini_start // mini_batch_size + 1) % grad_accum) == 0 or mini_start + mini_batch_size >= len(order):
                        torch.nn.utils.clip_grad_norm_(param_list, max_grad_norm)
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                if target_kl > 0.0 and epoch_kls:
                    mean_kl = sum(epoch_kls) / len(epoch_kls)
                    if mean_kl > target_kl:
                        break

            if epoch_kls:
                mean_kl = sum(epoch_kls) / len(epoch_kls)
                if mean_kl > target_kl * 1.5:
                    self._kl_coef = min(self._kl_coef * 1.5, 1.0)
                elif mean_kl < target_kl * 0.5:
                    self._kl_coef = max(self._kl_coef * 0.5, 1e-5)
                logging.info(f"Adaptive KL: mean_kl={mean_kl:.4f}, target_kl={target_kl:.4f}, new_kl_coef={self._kl_coef:.6f}")

            avg_loss = total_loss / max(updates, 1)

            logging.info(f"Training step completed: avg_loss={avg_loss:.4f}, total_updates={updates}, final_kl_coef={self._kl_coef:.6f}")

            with torch.no_grad():
                for ref_param, cur_param in zip(
                    self._reference_model.parameters(), self._model.parameters()
                ):
                    ref_param.data.mul_(0.99).add_(cur_param.data, alpha=0.01)
            self._weights_updated_at = time.time()
            if self._scheduler:
                self._scheduler.step()
            self.temperature *= float(training.temperature_decay)
            return float(avg_loss)

    def _encode_trajectory(self, traj: Trajectory) -> dict[str, Any]:
        torch = self._torch
        prompt = traj.task.prompt or ""
        response = traj.response or ""

        prompt_ids = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        full_ids = self._tokenizer(
            prompt + response,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]

        labels = full_ids.clone()
        prompt_len = int(prompt_ids.shape[1])
        labels[:, :prompt_len] = -100

        return {
            "input_ids": full_ids.to(self._model.device),
            "labels": labels.to(self._model.device),
            "prompt_len": prompt_len,
        }

    def _response_nll(self, input_ids: Any, labels: Any, *, use_reference: bool) -> Any:
        model = self._reference_model if use_reference else self._model
        out = model(input_ids=input_ids, labels=labels)
        return out.loss

    def _response_log_prob_sum(self, input_ids: Any, labels: Any, *, use_reference: bool) -> Any:
        model = self._reference_model if use_reference else self._model
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        mask = (labels != -100).to(dtype=loss.dtype)
        n = mask.sum().clamp(min=1)
        return -loss * n

    def _response_entropy(self, input_ids: Any, labels: Any, *, use_reference: bool) -> Any:
        torch = self._torch
        model = self._reference_model if use_reference else self._model
        out = model(input_ids=input_ids, labels=None)
        logits = out.logits  # (batch, seq_len, vocab)
        mask = (labels != -100)  # (batch, seq_len)
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        # Compute entropy per token
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        probs = torch.softmax(shift_logits, dim=-1)
        entropy_per_token = -torch.sum(probs * log_probs, dim=-1)  # (batch, seq_len-1)
        # Sum over response tokens
        entropy_sum = (entropy_per_token * shift_mask).sum(dim=-1)  # (batch,)
        return entropy_sum

    def _value_at_prompt(self, prompt: str, *, requires_grad: bool) -> Any:
        torch = self._torch
        if self._value_head is None:
            z = torch.zeros((), device=self._model.device, dtype=torch.float32)
            return z if requires_grad else z.detach()
        enc = self._tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx:
            out = self._model(**enc, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float()
            v = self._value_head(h).squeeze(-1).squeeze(0)
        return v

    def _start_server(self) -> None:
        try:
            import uvicorn
            from fastapi import FastAPI, Header, HTTPException
            from fastapi.responses import JSONResponse, StreamingResponse
        except ImportError as exc:
            raise RuntimeError("FastAPI/uvicorn are required for in-process model serving.") from exc

        app = FastAPI(title="SN851 Runtime LLM Server")
        runtime = self

        def _check_auth(auth_header: str | None) -> None:
            expected = f"Bearer {runtime.serve_api_key}"
            if runtime.serve_api_key and auth_header != expected:
                raise HTTPException(status_code=401, detail="Unauthorized")

        def _weights_meta() -> dict[str, Any]:
            ts = runtime._weights_updated_at
            if ts is None:
                return {"weights_updated_at": None, "weights_updated_at_iso": None}
            return {
                "weights_updated_at": ts,
                "weights_updated_at_iso": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            }

        @app.get("/health")
        async def health() -> dict[str, Any]:
            return {"ok": True, "model": runtime.model_name, **_weights_meta()}

        @app.get("/v1/models")
        async def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
            _check_auth(authorization)
            return {
                "object": "list",
                "data": [{"id": runtime.model_name, "object": "model", "owned_by": "sn851-local"}],
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(payload: dict[str, Any], authorization: str | None = Header(default=None)) -> Any:
            _check_auth(authorization)
            runtime._lazy_init()
            try:
                n_val = int(payload.get("n", 1))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="Invalid n.")
            if n_val != 1:
                raise HTTPException(status_code=400, detail="Only n=1 is supported.")

            messages = payload.get("messages", [])
            stream = bool(payload.get("stream", False))
            max_tokens = payload.get("max_tokens")
            max_new_tokens = int(max_tokens) if max_tokens is not None else None
            temp_raw = payload.get("temperature")
            temperature_f = float(temp_raw) if temp_raw is not None else None
            top_p_raw = payload.get("top_p")
            top_p_f = float(top_p_raw) if top_p_raw is not None else None
            seed_raw = payload.get("seed")
            seed_i = int(seed_raw) if seed_raw is not None else None
            stops = runtime._normalize_stop(payload.get("stop"))
            model_id = str(payload.get("model", runtime.model_name))
            req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            def _sync_complete() -> tuple[str, int, int, str]:
                with runtime._model_lock:
                    prompt = runtime._build_prompt_from_messages(messages)
                    return runtime._generate_unlocked_with_meta(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature_f,
                        top_p=top_p_f,
                        stop=stops,
                        seed=seed_i,
                    )

            if not stream:
                text, pt, ct, fr = await asyncio.to_thread(_sync_complete)
                usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
                return JSONResponse(
                    {
                        "id": req_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": fr,
                            }
                        ],
                        "usage": usage,
                    }
                )

            def event_stream() -> Generator[str, None, None]:
                runtime._lazy_init()
                header = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(header)}\n\n"
                with runtime._model_lock:
                    prompt = runtime._build_prompt_from_messages(messages)
                    for delta, is_final, usage, fr_reason in runtime._stream_chat_unlocked(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature_f,
                        top_p=top_p_f,
                        stop=stops,
                        seed=seed_i,
                    ):
                        if is_final:
                            tail = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": fr_reason or "stop",
                                    }
                                ],
                                "usage": usage,
                            }
                            yield f"data: {json.dumps(tail)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        if delta:
                            item = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": delta},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(item)}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        config = uvicorn.Config(
            app,
            host=self.serve_host,
            port=self.serve_port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._server_thread = threading.Thread(target=self._server.run, daemon=True)
        self._server_thread.start()

        deadline = time.time() + 10
        while time.time() < deadline:
            if getattr(self._server, "started", False):
                return
            time.sleep(0.05)
        raise RuntimeError("Failed to start in-process model server within timeout")
