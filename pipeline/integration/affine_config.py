import json
from urllib.request import urlopen
from typing import Dict, List, Tuple

from ..schemas import EnvConfig


def load_env_configs(
    *,
    system_config_url: str,
) -> Dict[str, EnvConfig]:
    environments = _load_environments_from_url(system_config_url)
    return _build_env_configs(environments)


def _load_environments_from_url(system_config_url: str) -> Dict[str, dict]:
    try:
        with urlopen(system_config_url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise ValueError("Failed to fetch system config from URL. Check pipeline.system_config_url.") from exc
    if isinstance(payload, dict) and isinstance(payload.get("environments"), dict):
        return payload["environments"]
    raise ValueError("Unexpected system_config JSON format from URL")


def _build_env_configs(
    environments: Dict[str, dict],
) -> Dict[str, EnvConfig]:
    result: Dict[str, EnvConfig] = {}

    for env_name, env_cfg in environments.items():
        sampling_cfg = env_cfg.get("sampling_config", {})
        sampling_list = sampling_cfg.get("sampling_list", [])
        result[env_name] = EnvConfig(
            env_name=env_name,
            enabled_for_sampling=bool(env_cfg.get("enabled_for_sampling", False)),
            enabled_for_scoring=bool(env_cfg.get("enabled_for_scoring", False)),
            min_completeness=float(env_cfg.get("min_completeness", 0.9)),
            scheduling_weight=float(sampling_cfg.get("scheduling_weight", 1.0)),
            sampling_list=[int(t) for t in sampling_list],
            sampling_count=int(sampling_cfg.get("sampling_count", len(sampling_list) or 0)),
            rotation_enabled=bool(sampling_cfg.get("rotation_enabled", True)),
            rotation_count=int(sampling_cfg.get("rotation_count", 0)),
            rotation_interval=int(sampling_cfg.get("rotation_interval", 3600)),
            display_name=env_cfg.get("display_name"),
        )

    return result


def split_active_envs(envs: Dict[str, EnvConfig]) -> Tuple[List[str], List[str]]:
    sampling = [k for k, v in envs.items() if v.enabled_for_sampling]
    scoring = [k for k, v in envs.items() if v.enabled_for_scoring]
    return sampling, scoring
