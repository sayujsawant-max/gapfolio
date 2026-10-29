-- Postgres-only bits that SQLAlchemy's create_all does not cover.
-- Applied by scripts/migrate.py, or by hand: psql "$DATABASE_URL" -f db/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- Approximate nearest-neighbour index for the semantic match. HNSW beats
-- IVFFlat for a read-heavy, low-write workload and needs no training step.
-- The dimension lives on the column (models.py, GAPFOLIO_EMBEDDING_DIM=384 for the default SBERT model); changing it means re-indexing every document.
CREATE INDEX IF NOT EXISTS ix_embeddings_vector_hnsw
    ON embeddings USING hnsw (vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Every search filters by owner before ranking, so at pilot scale this
-- composite does more work than the ANN index does.
CREATE INDEX IF NOT EXISTS ix_embeddings_owner_lookup
    ON embeddings (owner_type, owner_id);

CREATE INDEX IF NOT EXISTS ix_reports_user_created
    ON reports (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_consents_user_scope
    ON consents (user_id, scope, created_at DESC);

-- The retention sweep scans this nightly.
CREATE INDEX IF NOT EXISTS ix_resumes_purge
    ON resumes (purge_after) WHERE purge_after IS NOT NULL;
