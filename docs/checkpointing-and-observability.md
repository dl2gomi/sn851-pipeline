# Checkpointing, PostgreSQL registry, and Prometheus + Grafana

This document is the **design plan** for production-grade artifacts, metadata, and live telemetry. Implementation can be phased; the split of responsibilities below should stay stable.

## 1. Goals

- **Run identity:** Each pipeline invocation has a **`run_id`** (`PipelineOrchestrator.run_id`, from YAML `pipeline.run_id`, CLI `--run-id`, or a generated UUID). Use it as the foreign key for Postgres and as a Prometheus label.

- **Weights:** Full Hugging Face–style directories on disk (or object storage later), **no** automatic Hub upload from the pipeline.
- **Truth for humans and automation:** **PostgreSQL** holds run identity, step/checkpoint records, and pointers to weight paths—queryable and backup-friendly.
- **Live visibility:** **Prometheus** scrapes (or receives) low-cardinality training metrics; **Grafana** dashboards for real-time observation and alerts.
- **Training must not depend on observability:** If Postgres or Prometheus is down, the RL loop continues; failures are logged and optionally retried.

## 2. Three-layer architecture

| Layer | Technology | Holds |
|-------|------------|--------|
| **Data plane (large blobs)** | Local FS / S3-compatible later | `model.safetensors` shards, tokenizer, `config.json` |
| **Control plane (metadata)** | **PostgreSQL** | Runs, steps, checkpoint rows, config snapshot IDs, integrity hints |
| **Telemetry plane (time series)** | **Prometheus** (+ **Grafana**) | Gauges/counters for current (or few) runs—**not** raw weights |

Do not store multi-GB tensors in Postgres or Prometheus.

## 3. Checkpointing (full model write)

### 3.1 Trigger

Reuse existing **export gating**: `export_every_n_steps`, `eligible_for_scoring`, no hard per-env regression vs best, geometric mean strictly improves tracked best.

### 3.2 Filesystem layout

- Root: `artifacts_dir / checkpoints /` (configurable).
- Each export: `step_XXXXXX/` or `run_{uuid}_step_XXXXXX/` if multiple concurrent runs share one host.
- **Atomic write:** `step_XXXXXX.incomplete/` → rename to `step_XXXXXX/` after `save_pretrained` + `tokenizer.save_pretrained` succeed.
- **Retention:** `keep_last_n` full trees on disk; older directories removed after a successful new export (or moved to cold storage by an external job).

### 3.3 Manifest

Inside each checkpoint directory:

- `export_manifest.json`: step, `run_id`, geometric mean, per-env scores and completeness, `train_step_loss`, `weights_updated_at`, source `model_dir`, format version.

### 3.4 Postgres alignment

When a checkpoint is **committed** (after atomic rename):

- Insert or update row in `checkpoints` (see §4) with **absolute or URI path**, step, metrics snapshot, optional content hash of `config.json` + shard list (not full tensor hash unless you accept cost).

## 4. PostgreSQL schema (conceptual)

Use migrations (e.g. Alembic) in a real deployment. Types are indicative.

**`runs`**

- `id` (UUID PK), `created_at`, `finished_at`, `status` (`running` / `completed` / `failed`)
- `config_hash` or JSONB `config_snapshot` (pipeline + training knobs)
- `model_dir` (source HF folder), `artifacts_dir`, `git_commit` (optional)

**`orchestrator_steps`**

- `id` (bigserial), `run_id` (FK), `step` (int), `started_at`, `ended_at`
- `train_loss`, `avg_reward`, `avg_kl`, `geometric_mean`, `eligible_for_scoring` (bool)
- `env_scores` JSONB, `env_completeness` JSONB (bounded env count—OK)
- `exported_checkpoint` (bool) — whether this step produced a weight export
- `weights_updated_at` (timestamptz) — snapshot of `SharedModelRuntime.weights_updated_at` after this step’s train phase (same meaning as checkpoint rows; `NULL` if model never loaded)

**`checkpoints`**

- `id` (UUID PK), `run_id` (FK), `step` (int), `storage_path` (text) — directory path or `s3://…`
- `geometric_mean`, `env_scores` JSONB, `weights_updated_at` (timestamptz), `manifest_version` (int)
- `bytes_total` (optional), `shard_files` JSONB (optional list for audit)

**Indexes:** `(run_id, step)`, `run_id` on `checkpoints`, `created_at` on `runs`.

**Optional later:** `batches` table only if you need per–mini-batch analytics—high volume; prefer aggregates in `orchestrator_steps` first.

## 5. Prometheus metrics

### 5.1 Principles

- **Low cardinality:** Labels like `run_id` (one active run) or a **short** fixed `env` label set (`GAME`, `SWE-INFINITE`, …). Never use `task_id` or unbounded strings as labels.
- **Update cadence:** Once per **orchestrator step** is enough for RL; avoid per-token or per-batch unless you aggregate in-process.
- **Exposure:** `prometheus_client` HTTP `/metrics` on localhost (trainer process) **or** Pushgateway if the process cannot be scraped long-lived.

### 5.2 Suggested metric names (examples)

| Metric | Type | Labels | Notes |
|--------|------|--------|--------|
| `sn851_train_loss` | Gauge | `run_id` | Step-level loss |
| `sn851_geometric_mean` | Gauge | `run_id` | Eval GM |
| `sn851_eligible_for_scoring` | Gauge | `run_id` | 0 or 1 |
| `sn851_env_score` | Gauge | `run_id`, `env` | One series per env |
| `sn851_checkpoint_exported` | Counter | `run_id` | Increment when a full export succeeds |
| `sn851_orchestrator_step` | Gauge | `run_id` | Current step index |
| `sn851_weights_updated_timestamp` | Gauge | `run_id` | Unix time from runtime |

Process metrics (optional): `process_resident_memory_bytes`, GPU from `nvidia-smi` exporter or DCGM—separate from app metrics.

### 5.3 Grafana

- **Dashboards:** Training overview (loss, GM, eligibility); per-env score grid; checkpoint export events; step timeline.
- **Alerts (examples):** no `checkpoint_exported` increase for N hours while `eligible`; `train_loss` NaN; process up/down via blackbox or `up` metric.

## 6. End-to-end flow (one orchestrator step)

1. **Rollout → train → eval** (existing).
2. **Postgres:** insert `orchestrator_steps` row with metrics (transaction at end of step).
3. **Prometheus:** set gauges for this step (same values as logged).
4. **Export:** if `_maybe_export` passes gates, write weights atomically, then insert `checkpoints` row + increment `sn851_checkpoint_exported`, set `exported_checkpoint` on the step row.
5. **Grafana:** refreshes from Prometheus; operators see curves without querying Postgres for every plot (Postgres remains source of truth for audit and paths).

## 7. Failure and consistency

- **Postgres write fails:** log error; optionally buffer to local JSONL for replay; do not stop training.
- **Prometheus unreachable:** training continues; metrics gap is visible in Grafana as flat line / missing points.
- **Checkpoint write fails:** no atomic rename; no `checkpoints` row; alert on failed export if you add a `sn851_checkpoint_write_errors` counter.
- **Order:** persist weights to disk **before** inserting DB row that references the path, or insert with `status=pending` then finalize—avoid dangling DB rows without files.

## 8. Implementation status (in repo)

- **`run_id`:** CLI `--run-id` / YAML `pipeline.run_id` / UUID default; used in Postgres and Prometheus labels.
- **Checkpointing:** `SharedModelRuntime.save_hf_checkpoint`, `pipeline/core/scoring/checkpoint_fs.py` (atomic dir, prune, `index.jsonl`), `Exporter` writes JSON plus optional full HF tree under `artifacts_dir` / `checkpoint.subdir`.
- **PostgreSQL:** `pipeline/persistence/` — `protocols.py`, `noop.py`, `factory.py`, `postgres/schema.sql`, `postgres/repository.py`. Set `pipeline.postgres.dsn`; failures are logged and do not stop training.
- **Prometheus:** `pipeline/observability/` — `prometheus/recorder.py` binds `/metrics` when `pipeline.prometheus.enabled`; `factory.py` selects recorder vs no-op.

### Optional hardening

- Alembic migrations instead of DDL at connect; connection pooling; scrape config + Grafana samples in `deploy/` or docs.

## 9. Future revisions

- **Lossless delta / sparse checkpoints** (research-grade, bit-exact): revisit after full-write path is stable and disk cost is measured—see README note.
- **Object storage** for `storage_path` URIs instead of local paths only.
- **TimescaleDB** hypertable for `orchestrator_steps` if step volume explodes—optional extension to Postgres.
