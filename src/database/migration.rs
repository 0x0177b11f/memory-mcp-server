use diesel::prelude::*;
use diesel::sql_query;
use tracing::info;

use super::Database;

const MIGRATION_V020_ADD_EMBEDDINGS: &str = "ALTER TABLE documents
     ADD COLUMN IF NOT EXISTS name_embedding vector(384),
     ADD COLUMN IF NOT EXISTS description_embedding vector(384)";

const MIGRATION_V020_DROP_NAME_UNIQUE: &str =
    "ALTER TABLE IF EXISTS documents DROP CONSTRAINT IF EXISTS documents_name_key;";

fn drop_old_indexes_v020(conn: &mut PgConnection) -> anyhow::Result<()> {
    let index_statements = [
        "DROP INDEX IF EXISTS documents_name_embedding_idx",
        "DROP INDEX IF EXISTS documents_description_embedding_idx",
        "DROP INDEX IF EXISTS documents_name_trgm_idx",
        "DROP INDEX IF EXISTS documents_description_trgm_idx",
        "DROP INDEX IF EXISTS documents_description_coalesce_trgm_idx",
    ];
    for statement in index_statements {
        sql_query(statement).execute(conn)?;
    }
    Ok(())
}

fn recreate_indexes_v020(conn: &mut PgConnection) -> anyhow::Result<()> {
    let index_statements = [
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
    ];
    for statement in index_statements {
        sql_query(statement).execute(conn)?;
    }
    Ok(())
}

fn run_migration_v020(conn: &mut PgConnection) -> anyhow::Result<()> {
    sql_query(MIGRATION_V020_ADD_EMBEDDINGS).execute(conn)?;
    sql_query(MIGRATION_V020_DROP_NAME_UNIQUE).execute(conn)?;
    drop_old_indexes_v020(conn)?;
    recreate_indexes_v020(conn)?;
    Ok(())
}

impl Database {
    pub fn migrate_database(&self) -> anyhow::Result<()> {
        info!("Running database migrations");
        let mut conn = self.get_conn()?;

        // 0.2.0 migration
        run_migration_v020(&mut conn)?;

        info!("Database migrations complete");
        Ok(())
    }
}
