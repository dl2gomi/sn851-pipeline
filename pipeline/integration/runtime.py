from __future__ import annotations

import asyncio
import copy
import json
import threading
import time
import uuid
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
        self.temperature = temperature
        self.top_p = top_p
        self.serve_host = serve_host
        self.serve_port = serve_port
        self.serve_api_key = serve_api_key

        self._tokenizer: Any = None
        self._model: Any = None
        self._reference_model: Any = None
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

        # Reference model for KL regularization anchor (frozen snapshot).
        self._reference_model = copy.deepcopy(self._model).eval()
        for p in self._reference_model.parameters():
            p.requires_grad = False

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=float(self.training_defaults.lr),
        )

        self._weights_updated_at = time.time()

    def generate(self, task: Task) -> str:
        return self.generate_from_prompt(task.prompt or "")

    def generate_from_prompt(self, prompt: str) -> str:
        self._lazy_init()
        with self._model_lock:
            return self._generate_unlocked(prompt=prompt)

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

        with self._model_lock:
            grad_accum = max(1, int(training.gradient_accumulation_steps))
            ppo_epochs = max(1, int(training.ppo_epochs))
            kl_coef = float(training.kl_coef)
            max_grad_norm = float(training.max_grad_norm)

            rewards = torch.tensor([float(t.reward) for t in batch], device=self._model.device)
            reward_baseline = rewards.mean()
            advantages = rewards - reward_baseline

            encoded = [self._encode_trajectory(t) for t in batch]
            total_loss = 0.0
            updates = 0
            self._optimizer.zero_grad(set_to_none=True)

            for _ in range(ppo_epochs):
                for i, item in enumerate(encoded):
                    nll_cur = self._response_nll(item["input_ids"], item["labels"], use_reference=False)
                    with torch.no_grad():
                        nll_ref = self._response_nll(item["input_ids"], item["labels"], use_reference=True)

                    adv = advantages[i].detach()
                    policy_loss = adv * nll_cur
                    kl_loss = (nll_cur - nll_ref) ** 2
                    loss = policy_loss + (kl_coef * kl_loss)
                    (loss / grad_accum).backward()
                    total_loss += float(loss.detach().item())
                    updates += 1

                    if ((i + 1) % grad_accum) == 0 or (i + 1) == len(encoded):
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_grad_norm)
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

            avg_loss = total_loss / max(updates, 1)

            with torch.no_grad():
                for ref_param, cur_param in zip(self._reference_model.parameters(), self._model.parameters()):
                    ref_param.data.mul_(0.99).add_(cur_param.data, alpha=0.01)
            self._weights_updated_at = time.time()
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

    def _generate_unlocked(self, prompt: str) -> str:
        torch = self._torch
        model_inputs = self._tokenizer(prompt, return_tensors="pt")
        model_inputs = {k: v.to(self._model.device) for k, v in model_inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        completion_ids = out[0][model_inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(completion_ids, skip_special_tokens=True)

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
            messages = payload.get("messages", [])
            stream = bool(payload.get("stream", False))
            prompt = "\n".join(
                str(m.get("content", "")) for m in messages if isinstance(m, dict) and m.get("content")
            )
            text = runtime.generate_from_prompt(prompt)

            req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": len(text.split()),
                "total_tokens": len(text.split()),
            }

            if not stream:
                return JSONResponse(
                    {
                        "id": req_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": payload.get("model", runtime.model_name),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": usage,
                    }
                )

            def event_stream() -> Generator[str, None, None]:
                chunks = text.split() if text else [""]
                header = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": payload.get("model", runtime.model_name),
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(header)}\n\n"
                for idx, token in enumerate(chunks):
                    delta = token if idx == len(chunks) - 1 else f"{token} "
                    item = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": payload.get("model", runtime.model_name),
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(item)}\n\n"
                tail = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": payload.get("model", runtime.model_name),
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": usage,
                }
                yield f"data: {json.dumps(tail)}\n\n"
                yield "data: [DONE]\n\n"

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
