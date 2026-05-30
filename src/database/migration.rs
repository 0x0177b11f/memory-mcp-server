use diesel::prelude::*;
use diesel::sql_query;
use tracing::info;

use super::Database;

const MIGRATION_V020_ADD_EMBEDDINGS: &str = "ALTER TABLE documents
     ADD COLUMN IF NOT EXISTS name_embedding vector(384),
     ADD COLUMN IF NOT EXISTS description_embedding vector(384)";

const MIGRATION_V020_DROP_NAME_UNIQUE: &str =
    "ALTER TABLE IF EXISTS documents DROP CONSTRAINT IF EXISTS documents_name_key;";

fn run_migration_v020(conn: &mut PgConnection) -> anyhow::Result<()> {
    sql_query(MIGRATION_V020_ADD_EMBEDDINGS).execute(conn)?;
    sql_query(MIGRATION_V020_DROP_NAME_UNIQUE).execute(conn)?;
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
