# SN851 RL Pipeline

This folder contains an Affine-aligned RL optimization control-plane skeleton for SN851.

- loads environment config from public `system_config.json` URL
- samples tasks from env `sampling_list` with scheduling weights
- maps reward directly from environment score
- evaluates with scorer-style geometric mean and per-env completeness gates
- exports best checkpoints only when scoring eligibility is satisfied
- enforces Affine Qwen3-32B architecture guardrails for the local model folder
- applies Affine-style sampling rate limiting (rotation-aware + 48h minimum completion guarantee)

The shared in-process runtime is implemented in `pipeline/integration/runtime.py` (lazy model load, generation, reward-weighted update, in-process OpenAI-compatible `/v1` for env callbacks).
`pipeline/integration/environment_executor.py` wires **affinetes** docker envs via `env_images.json`.
We currently **assume all configured envs are healthy** (images pull, `evaluate` returns scores, LLM callbacks reach the local server). Re-validate after affinetes or image updates; see **Environment integration (re-check later)** below.
The runtime is designed for in-process RL with separation of concerns: policy and trainer are separate adapters that share one model runtime.

## Central configuration

All pipeline settings are centralized in root `configuration.yaml`.

- default path: `sn851-pipeline/configuration.yaml`
- override path: `python -m pipeline.main --config <path> ...`
- CLI flags still work and override the file for that run

Key RL knobs in `configuration.yaml`:
- `pipeline.training.rollout_batch_size`
- `pipeline.training.mini_batch_size`
- `pipeline.training.gradient_accumulation_steps`
- `pipeline.training.ppo_epochs`
- `pipeline.training.clip_range`
- `pipeline.training.target_kl`
- `pipeline.training.kl_coef`
- `pipeline.training.use_value_head`
- `pipeline.training.value_loss_coef`

Environment config knob:
- `pipeline.system_config_url`: URL of Affine `system_config.json` used by the pipeline
- `pipeline.serve.host|port|api_key`: in-process OpenAI-compatible server for env-side inference callbacks
- `pipeline.postgres.enabled|dsn`: optional external PostgreSQL for run/step/checkpoint registry
- `pipeline.prometheus.enabled|host|port`: optional in-process `/metrics` for Grafana scrape targets
- `pipeline.checkpoint.*`: full HF-style weight export (subdir, retention, atomic writes) — see `docs/checkpointing-and-observability.md`

## Layout

- `pipeline/app/` - CLI and orchestration entrypoints
- `pipeline/core/` - core runtime components split by domain
- `pipeline/integration/` - Affine/system integration adapters
- `pipeline/config/` - pipeline/training/scoring config dataclasses
- `pipeline/schemas/` - shared datatypes for tasks, trajectories, and metrics
- `pipeline/integration/affine_config.py` - loads env config from Affine `system_config.json`
- `pipeline/integration/runtime.py` - shared in-memory model runtime
- `pipeline/integration/runtime.py` also runs an in-process FastAPI `/v1` LLM endpoint bound to the same model weights
- `pipeline/integration/policy.py` - policy adapter (inference only)
- `pipeline/integration/trainer_backend.py` - trainer adapter (updates only)
- `pipeline/integration/environment_executor.py` - environment evaluation adapter
- `pipeline/integration/backends.py` - compatibility re-exports of integration adapters
- `pipeline/integration/model_guard.py` - Qwen3-32B architecture validation (Affine-compatible)
- `pipeline/core/sampling/sampler.py` - sampling-list and weighted task sampler
- `pipeline/core/execution/rollout.py` - policy + environment executor interfaces and rollout worker
- `pipeline/core/scoring/reward.py` - Affine-style scalar reward shaping
- `pipeline/core/state/replay_buffer.py` - in-memory replay buffer
- `pipeline/core/training/trainer.py` - training controller using a real backend adapter
- `pipeline/core/scoring/evaluator.py` - geometric-mean and completeness evaluator
- `pipeline/core/scoring/exporter.py` - gated export (metrics JSON + optional full model tree)
- `pipeline/core/scoring/checkpoint_fs.py` - atomic checkpoint dirs, prune, `index.jsonl`
- `pipeline/persistence/` - Postgres registry (`protocols`, `factory`, `postgres/repository`, `noop`)
- `pipeline/observability/` - Prometheus metrics (`factory`, `prometheus/recorder`, `noop`)
- `pipeline/config/snapshot.py` - JSON-safe pipeline config snapshot for DB rows
- `pipeline/app/orchestrator.py` - end-to-end control loop
- `pipeline/app/cli.py` - CLI argument parsing and run wiring
- `pipeline/main.py` - CLI entrypoint
- `configuration.yaml` - centralized runtime and CLI defaults

## Preflight (works now)

From `E:\Works\Main\SN851\beta1\sn851-pipeline`:

```bash
python -m pipeline.main run --steps 10 --dry-run --rollouts-per-step 32 --batch-size 16
```
`--dry-run` performs only preflight checks (model config + environment config).

Backward-compatible usage without `run` also works:
```bash
python -m pipeline.main --steps 10 --dry-run
```

## CLI commands (`python -m pipeline.main`)

### `run`

- `--steps` (int, default: `5`)
  - Number of orchestrator steps to run in non-dry mode.
- `--dry-run` (flag)
  - Run only preflight checks and exit before rollout/training.
- `--rollouts-per-step` (int, default: `16`)
  - Override for rollout batch size in this run.
- `--batch-size` (int, default: `8`)
  - Override for mini-batch size in this run.
- `--model-dir` (path, optional)
  - Folder containing model `config.json`.
  - If omitted, defaults to `model/` under the project root.
- `--train-all-sampling-envs` (flag)
  - If set, train on all `enabled_for_sampling` environments.
  - If not set, train only on `enabled_for_scoring` environments.
- `--run-id` (string, optional)
  - Stable identifier for this process (checkpoint registry, Postgres, Prometheus). If omitted, a random UUID is assigned and stored on `PipelineConfig.run_id`.
- `--config` (path, global)
  - Path to central `configuration.yaml`.
  - Default is `configuration.yaml` in project root.

### `pull-model`

- `uid` (required)
  - Miner UID to pull from using Affine CLI (`af pull`).
- `--model-dir` (path, optional)
  - Output directory for pulled model.
  - Defaults to `model/` under this project.

## Runtime modes

- `dry-run`:
  - validates `--model-dir/config.json` presence
  - validates Qwen3-32B compatibility
  - validates env config loading
  - exits before rollout/training
- non-dry run:
  - executes orchestrator loop: rollouts, replay, `train_step`, scoring/export hooks
  - uses `SharedModelRuntime`, policy/trainer adapters, and `AffineEnvironmentExecutor` (affinetes)

## Concurrency and async (sync-first)

- **Training loop** (`pipeline/app/orchestrator.py`): synchronous on the main thread — each step does collect → replay → `train_step` → eval in order. There is no overlapping RL update from multiple workers.
- **Rollouts** (`pipeline/core/execution/rollout.py`): tasks are handled **one after another** in a plain loop (no parallel rollout workers in this codebase).
- **Shared weights** (`pipeline/integration/runtime.py`): a **threading `RLock`** serializes `train_step` and `generate*` on the same in-memory model, so two training steps never run concurrently on one runtime, and inference waits while a train step holds the lock (and vice versa).
- **In-process LLM server**: FastAPI/Uvicorn runs in a **background thread** with async route handlers; callbacks into `generate_from_prompt` still contend on that same lock. This is for OpenAI-compatible HTTP, not for overlapping GPU updates.
- **Affinetes / env executor** (`pipeline/integration/environment_executor.py`): affinetes is async internally; the adapter bridges from sync `evaluate()` using `asyncio.run` per call. That is intentional: keep the pipeline **sync** at the orchestration boundary. Revisit a dedicated async event loop or parallel rollouts **only if** profiling shows a hard need (throughput, not speculative rewrites).
- **Later — async / overlapping training**: If wall-clock cost becomes limiting, re-evaluate **async or pipelined training** (e.g. overlapping collection and updates, richer asyncio boundaries). Treat that as a deliberate design pass: off-policy staleness, concurrent weight writes, and lock narrowing all affect stability; do not adopt “async training” as a blind speed-up without aligning the RL loop and measuring.

## Environment integration (re-check later)

Assumption today: envs in `env_images.json` match `system_config.json` names, Docker can run them, and `evaluate(...)` is compatible with the kwargs passed from `environment_executor.py`. When upgrading affinetes, images, or the serve URL/host strategy, spot-check:

- One rollout per env class you care about; confirm non-zero scores when appropriate.
- Container can reach the pipeline host at `pipeline.serve.host:port` (not only `127.0.0.1` if Docker networking differs).
- `/health` shows `weights_updated_at` advancing after training steps.

**Later — per-env validation**: Add systematic checks beyond ad-hoc smoke tests: for each docker image / env name, document required `evaluate(...)` kwargs, timeouts, and LLM expectations; optional CI or a “preflight env matrix” that runs one short evaluation per env so regressions surface when images or affinetes change.

## Train from top Affine model

1. Download/pull the full model snapshot (60+ GB is expected) and keep it anywhere on disk.
2. Start training loop with that path:
```bash
python -m pipeline.main run --steps 100 --rollouts-per-step 64 --batch-size 32 --model-dir "D:\models\top-affine-model"
```
The pipeline enforces Affine-required Qwen3-32B-compatible architecture from `config.json`.

By default, training focuses on `enabled_for_scoring` envs (Affine incentive path).
Use `--train-all-sampling-envs` to include sampling-only environments.

## Isolation mode (no local affine repo)

The pipeline uses this URL by default:
`https://raw.githubusercontent.com/AffineFoundation/affine-cortex/refs/heads/main/affine/database/system_config.json`

## Optional model pull (separate subcommand)

If you choose to pull through CLI:
```bash
python -m pipeline.main pull-model 42 --model-dir model
```
This runs `af pull 42 --model-path model` under the hood.

## Checkpoint export (full model write — planned)

**Production plan (PostgreSQL registry + Prometheus + Grafana + full weight dirs):** see [`docs/checkpointing-and-observability.md`](docs/checkpointing-and-observability.md).

**Current behavior:** `Exporter.export_if_best` only writes a small JSON manifest under `artifacts_dir` when gates pass. **Full weight files are not written yet.**

**Adopted strategy (for implementation):** **full Hugging Face–style directory** per successful export — `model.save_pretrained` + `tokenizer.save_pretrained` into a dedicated folder (multi-shard `safetensors` as HF does for large models). **No automatic Hugging Face Hub upload**; you upload the saved folder yourself after training.

**Gating (unchanged):** same as today — only on `export_every_n_steps` cadence, and only when `eligible_for_scoring`, no hard per-env regression vs the running best, and geometric mean **strictly improves** the tracked best.

**Operational plan:**

1. **Layout:** e.g. `artifacts/checkpoints/step_XXXXXX/` containing full model + tokenizer + `export_manifest.json` (step, GM, env scores, completeness, `train_step_loss`, `weights_updated_at`, source `model_dir`).
2. **Concurrency:** save under the same `SharedModelRuntime` lock as train/infer; briefly run the module in eval mode for export, then restore train mode.
3. **Atomicity:** write into `step_XXXXXX.incomplete`, then rename to `step_XXXXXX` so a crash mid-write does not masquerade as a good checkpoint.
4. **Retention:** config knob `keep_last_n` — delete older `step_*` trees after a successful export (large models need this).
5. **Audit:** optional append-only `artifacts/checkpoints/index.jsonl` (one line per export).

**Checkpointing strategy — revisit later:** Disk cost is large for 32B-class models. After we have real export metrics (time, disk, cadence), we may **update this strategy** — e.g. lossless compressed bundles, or research-style **sparse / delta** checkpoints **if** they are proven **bit-exact** against merged weights for our training loop. Until then, **full-model writes** are the reference approach.

## Project files

- `requirements.txt` - compatibility dependency file for tools that expect it
- `pyproject.toml` - Python project metadata (standalone, not tied to Affine repo lockfiles)
- `.gitignore` includes `model/` so large checkpoints are not committed

## Integrate your real stack

1. Tune/extend `SharedModelRuntime` in `pipeline/integration/runtime.py` for your exact PPO/GRPO strategy and distributed stack.
2. Keep `AffinePolicyAdapter` and `AffineTrainerBackend` as separate SoC wrappers over `SharedModelRuntime`.
3. Env execution: affinetes + `env_images.json` is wired; revisit the checklist in **Environment integration (re-check later)** when the stack changes.
4. Keep RL updates in `train_step()` without server restarts/reloads.
5. Implement full-model checkpoint export per **Checkpoint export (full model write — planned)**; Hub upload remains manual.
6. Wire Postgres, Prometheus/Grafana, and checkpoint export per [`docs/checkpointing-and-observability.md`](docs/checkpointing-and-observability.md).

Non-dry runs print per-step logs with:
- per-env scores
- per-env completeness
- scoring eligibility
- geometric-mean metric
- checkpoint export decision
