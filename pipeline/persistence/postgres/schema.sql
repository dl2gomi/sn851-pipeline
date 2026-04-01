-- Applied once at startup (CREATE IF NOT EXISTS). External PostgreSQL process.

CREATE TABLE IF NOT EXISTS runs (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    config_snapshot JSONB,
    model_dir TEXT NOT NULL,
    artifacts_dir TEXT NOT NULL,
    git_commit TEXT
);

CREATE TABLE IF NOT EXISTS orchestrator_steps (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step INT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ NOT NULL,
    train_loss DOUBLE PRECISION,
    avg_reward DOUBLE PRECISION,
    avg_kl DOUBLE PRECISION,
    geometric_mean DOUBLE PRECISION,
    eligible_for_scoring BOOLEAN NOT NULL,
    env_scores JSONB,
    env_completeness JSONB,
    exported_checkpoint BOOLEAN NOT NULL DEFAULT FALSE,
    weights_updated_at TIMESTAMPTZ,
    UNIQUE (run_id, step)
);

CREATE INDEX IF NOT EXISTS idx_orch_steps_run ON orchestrator_steps(run_id);
CREATE INDEX IF NOT EXISTS idx_orch_steps_run_step ON orchestrator_steps(run_id, step);

CREATE TABLE IF NOT EXISTS checkpoints (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step INT NOT NULL,
    storage_path TEXT NOT NULL,
    geometric_mean DOUBLE PRECISION,
    env_scores JSONB,
    weights_updated_at TIMESTAMPTZ,
    manifest_version INT NOT NULL DEFAULT 1,
    bytes_total BIGINT,
    shard_files JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cp_run ON checkpoints(run_id);
