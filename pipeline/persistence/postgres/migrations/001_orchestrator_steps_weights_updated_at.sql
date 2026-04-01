-- Run once on existing databases created before weights_updated_at on orchestrator_steps.
ALTER TABLE orchestrator_steps
    ADD COLUMN IF NOT EXISTS weights_updated_at TIMESTAMPTZ;
