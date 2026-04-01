"""Filesystem operations for full HF checkpoint trees (atomic dir, prune, index)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ...integration.runtime import SharedModelRuntime

logger = logging.getLogger(__name__)


def _parse_step_dir(name: str) -> int:
    if name.endswith(".incomplete"):
        name = name[: -len(".incomplete")]
    if name.startswith("step_"):
        return int(name.split("_", 1)[1])
    return -1


def directory_size_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def list_weight_artifacts(root: Path) -> List[str]:
    names: List[str] = []
    for pattern in ("*.safetensors", "*.bin", "*.pt"):
        for p in root.glob(pattern):
            names.append(p.name)
    return sorted(names)


def prune_old_checkpoints(checkpoints_root: Path, keep_last_n: int) -> None:
    if keep_last_n < 1:
        return
    dirs = [
        p
        for p in checkpoints_root.iterdir()
        if p.is_dir() and p.name.startswith("step_") and not p.name.endswith(".incomplete")
    ]
    dirs.sort(key=lambda p: _parse_step_dir(p.name), reverse=True)
    for d in dirs[keep_last_n:]:
        try:
            shutil.rmtree(d)
            logger.info("Pruned old checkpoint directory %s", d)
        except OSError:
            logger.exception("Failed to prune %s", d)


def append_index_jsonl(checkpoints_root: Path, payload: Dict[str, Any]) -> None:
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    path = checkpoints_root / "index.jsonl"
    line = json.dumps(payload, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def save_checkpoint_atomic(
    runtime: "SharedModelRuntime",
    *,
    checkpoints_root: Path,
    step: int,
    run_id: str,
    manifest: Dict[str, Any],
    atomic: bool,
    max_shard_size: str,
    safe_serialization: bool,
    keep_last_n: int,
) -> Path:
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    final = checkpoints_root / f"step_{step:06d}"
    incomplete = checkpoints_root / f"step_{step:06d}.incomplete"

    def _write_tree(target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        meta = runtime.save_hf_checkpoint(
            target,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )
        manifest_path = target / "export_manifest.json"
        full_manifest = {**manifest, **meta, "run_id": run_id, "step": step}
        manifest_path.write_text(json.dumps(full_manifest, indent=2, default=str), encoding="utf-8")

    if atomic:
        if incomplete.exists():
            shutil.rmtree(incomplete, ignore_errors=True)
        incomplete.mkdir(parents=True)
        _write_tree(incomplete)
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        incomplete.rename(final)
        out = final
    else:
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        final.mkdir(parents=True)
        _write_tree(final)
        out = final

    bytes_total = directory_size_bytes(out)
    shard_files = list_weight_artifacts(out)
    append_index_jsonl(
        checkpoints_root,
        {
            "run_id": run_id,
            "step": step,
            "path": str(out.resolve()),
            "bytes_total": bytes_total,
            "geometric_mean": manifest.get("geometric_mean"),
        },
    )
    prune_old_checkpoints(checkpoints_root, keep_last_n)
    return out
