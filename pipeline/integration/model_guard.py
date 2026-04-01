import json
from pathlib import Path

REQUIRED_MODEL_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "intermediate_size": 25600,
    "vocab_size": 151936,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
}


def validate_local_config(config_path: Path) -> tuple[bool, str]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for field, expected in REQUIRED_MODEL_CONFIG.items():
        actual = cfg.get(field)
        if actual != expected:
            return False, f"model_not_allowed:{field}={actual} (expected {expected})"
    return True, "ok"
