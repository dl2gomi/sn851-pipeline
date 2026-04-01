import asyncio
import json
import threading
from pathlib import Path
from typing import Any

from ..core.execution.rollout import EnvResult, EnvironmentExecutor
from ..schemas import Task
from .runtime import SharedModelRuntime


class AffineEnvironmentExecutor(EnvironmentExecutor):
    def __init__(self, runtime: SharedModelRuntime, env_images_path: Path | None = None):
        self.runtime = runtime
        self.env_images_path = env_images_path or Path(__file__).resolve().parents[2] / "env_images.json"
        self._env_images = self._load_env_images(self.env_images_path)
        self._env_instances: dict[str, Any] = {}
        self._lock = threading.Lock()

    def evaluate(self, task: Task, response: str) -> EnvResult:
        del response  # Affinetes envs call model via runtime base_url.
        try:
            result = asyncio.run(self._evaluate_async(task))
        except Exception:
            return EnvResult(score=0.0, kl_estimate=0.0, is_timeout=True, is_format_valid=False)

        score = float(result.get("score", 0.0))
        is_timeout = bool(result.get("timeout", False))
        is_format_valid = bool(result.get("is_format_valid", True))
        return EnvResult(
            score=score,
            kl_estimate=0.0,
            is_timeout=is_timeout,
            is_format_valid=is_format_valid,
        )

    async def close(self) -> None:
        with self._lock:
            items = list(self._env_instances.items())
            self._env_instances.clear()
        for _, env in items:
            try:
                await env.cleanup()
            except Exception:
                pass

    async def _evaluate_async(self, task: Task) -> dict[str, Any]:
        env = self._get_or_create_env(task.env)
        result = await env.evaluate(
            model=self.runtime.model_name,
            base_url=self.runtime.base_url,
            api_key=self.runtime.serve_api_key,
            task_id=int(task.task_id),
            temperature=self.runtime.temperature,
            _timeout=1800,
        )
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if isinstance(result, dict):
            return result
        return {"score": 0.0}

    def _get_or_create_env(self, env_name: str) -> Any:
        with self._lock:
            if env_name in self._env_instances:
                return self._env_instances[env_name]
            image = self._env_images.get(env_name)
            if not image:
                raise ValueError(f"No docker image mapping for env '{env_name}' in {self.env_images_path}")
            try:
                import affinetes as af_env
            except ImportError as exc:
                raise RuntimeError("affinetes is required for environment execution") from exc

            env = af_env.load_env(
                image=image,
                mode="docker",
                replicas=1,
                pull=True,
                cleanup=False,
            )
            self._env_instances[env_name] = env
            return env

    @staticmethod
    def _load_env_images(path: Path) -> dict[str, str]:
        if not path.exists():
            raise ValueError(f"env images map not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in raw.items()}
