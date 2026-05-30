use diesel::prelude::*;
use diesel::sql_query;
use tracing::{debug, info};

use super::Database;

const EXTENSIONS: [&str; 2] = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
];

const TABLES: [&str; 2] = [
    "CREATE TABLE IF NOT EXISTS documents (
        id bigserial PRIMARY KEY,
        name TEXT NOT NULL,
        name_embedding vector(384),
        description TEXT,
        description_embedding vector(384),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )",
    "CREATE TABLE IF NOT EXISTS memory_items (
        id bigserial PRIMARY KEY,
        document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        summary TEXT NOT NULL,
        summary_embedding vector(384),
        content TEXT NOT NULL,
        content_embedding vector(384),
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )",
];

const INDEXES: [&str; 11] = [
    "CREATE INDEX IF NOT EXISTS documents_name_embedding_idx
     ON documents USING hnsw (name_embedding vector_ip_ops)",
    "CREATE INDEX IF NOT EXISTS documents_description_embedding_idx
     ON documents USING hnsw (description_embedding vector_ip_ops)",
    "CREATE INDEX IF NOT EXISTS documents_name_trgm_idx
     ON documents USING gin (name gin_trgm_ops)",
     "CREATE INDEX IF NOT EXISTS documents_description_trgm_idx
     ON documents USING gin (description gin_trgm_ops)",
     "CREATE INDEX IF NOT EXISTS documents_description_coalesce_trgm_idx 
      ON documents USING gin (COALESCE(description, '') gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS memory_items_document_id_idx
     ON memory_items (document_id)",
    "CREATE INDEX IF NOT EXISTS memory_items_summary_embedding_idx
     ON memory_items USING hnsw (summary_embedding vector_ip_ops)",
    "CREATE INDEX IF NOT EXISTS memory_items_content_embedding_idx
     ON memory_items USING hnsw (content_embedding vector_ip_ops)",
    "CREATE INDEX IF NOT EXISTS memory_items_metadata_idx
     ON memory_items USING gin (metadata)",
    "CREATE INDEX IF NOT EXISTS memory_items_summary_trgm_idx
     ON memory_items USING gin (summary gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS memory_items_content_trgm_idx
     ON memory_items USING gin (content gin_trgm_ops)",
];

fn execute_statements(conn: &mut PgConnection, statements: &[&str]) -> anyhow::Result<()> {
    for statement in statements {
        sql_query(*statement).execute(conn)?;
    }
    Ok(())
}

impl Database {
    pub fn setup_database(&self) -> anyhow::Result<()> {
        info!("Setting up database");

        debug!("Creating extensions if not exist");
        let mut conn = self.get_conn()?;

        execute_statements(&mut conn, &EXTENSIONS)?;

        debug!("Creating documents table");
        debug!("Creating memory_items table");
        execute_statements(&mut conn, &TABLES)?;

        debug!("Creating indexes");
        execute_statements(&mut conn, &INDEXES)?;

        info!("Database setup complete");
        Ok(())
    }
}
