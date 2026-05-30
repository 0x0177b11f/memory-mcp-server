pub mod schema;
pub mod models;

#[cfg(test)]
mod tests;

use diesel::prelude::*;
use diesel::sql_query;
use diesel::sql_types::*;
use pgvector::Vector;

use self::models::*;
use tracing::{info, debug};

#[derive(Clone)]
pub struct Database {
    pool: diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>,
}

impl Database {
    pub fn new(database_url: &str) -> anyhow::Result<Self> {
        let manager = diesel::r2d2::ConnectionManager::<PgConnection>::new(database_url);
        let pool = diesel::r2d2::Pool::builder().build(manager)?;

        Ok(Self { pool })
    }

    pub fn get_conn(&self) -> anyhow::Result<diesel::r2d2::PooledConnection<diesel::r2d2::ConnectionManager<PgConnection>>> {
        Ok(self.pool.get()?)
    }

    pub fn setup_database(&self) -> anyhow::Result<()> {
        info!("Setting up database");

        debug!("Creating extensions if not exist");
        let mut conn = self.get_conn()?;

        sql_query("CREATE EXTENSION IF NOT EXISTS vector").execute(&mut conn)?;
        sql_query("CREATE EXTENSION IF NOT EXISTS pg_trgm").execute(&mut conn)?;

        debug!("Creating documents table");
        sql_query(
            "CREATE TABLE IF NOT EXISTS documents (
                id bigserial PRIMARY KEY,
                name TEXT NOT NULL,
                name_embedding vector(384),
                description TEXT,
                description_embedding vector(384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"
        ).execute(&mut conn)?;

        debug!("Creating memory_items table");
        sql_query(
            "CREATE TABLE IF NOT EXISTS memory_items (
                id bigserial PRIMARY KEY,
                document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                summary TEXT NOT NULL,
                summary_embedding vector(384),
                content TEXT NOT NULL,
                content_embedding vector(384),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"
        ).execute(&mut conn)?;

        debug!("Creating indexes");

        sql_query(
            "CREATE INDEX IF NOT EXISTS documents_name_embedding_idx
             ON documents USING hnsw (name_embedding vector_cosine_ops)"
        ).execute(&mut conn)?;

        sql_query(
            "CREATE INDEX IF NOT EXISTS documents_description_embedding_idx 
             ON documents USING hnsw (description_embedding vector_cosine_ops)"
        ).execute(&mut conn)?;

        sql_query(
            "CREATE INDEX IF NOT EXISTS documents_name_trgm_idx 
             ON documents USING gin (name gin_trgm_ops)"
        ).execute(&mut conn)?;

        sql_query(
            "CREATE INDEX IF NOT EXISTS documents_description_trgm_idx 
             ON documents USING gin (description gin_trgm_ops)"
        ).execute(&mut conn)?;

        sql_query(
            "CREATE INDEX IF NOT EXISTS memory_items_summary_embedding_idx
             ON memory_items USING hnsw (summary_embedding vector_cosine_ops)"
        ).execute(&mut conn)?;

        sql_query(
            "CREATE INDEX IF NOT EXISTS memory_items_content_embedding_idx 
             ON memory_items USING hnsw (content_embedding vector_cosine_ops)"
        ).execute(&mut conn)?;

        sql_query(
            "CREATE INDEX IF NOT EXISTS memory_items_metadata_idx 
             ON memory_items USING gin (metadata)"
        ).execute(&mut conn)?;

        sql_query(
            "CREATE INDEX IF NOT EXISTS memory_items_summary_trgm_idx 
             ON memory_items USING gin (summary gin_trgm_ops)"
        ).execute(&mut conn)?;

        sql_query(
            "CREATE INDEX IF NOT EXISTS memory_items_content_trgm_idx 
             ON memory_items USING gin (content gin_trgm_ops)"
        ).execute(&mut conn)?;

        self.migrate_database()?;
        info!("Database setup complete");
        Ok(())
    }

    pub fn migrate_database(&self) -> anyhow::Result<()> {
        info!("Running database migrations");
        let mut conn = self.get_conn()?;

        // 0.2.0 migration
        //// Add embedding columns to documents table
        sql_query(
            "ALTER TABLE documents 
             ADD COLUMN IF NOT EXISTS name_embedding vector(384), 
             ADD COLUMN IF NOT EXISTS description_embedding vector(384)"
        ).execute(&mut conn)?;

        //// Remove documents.name UNIQUE constraint if exists
        sql_query(
            "ALTER TABLE IF EXISTS documents DROP CONSTRAINT IF EXISTS documents_name_key;"
        ).execute(&mut conn)?;

        info!("Database migrations complete");
        Ok(())
    }

    pub fn create_document(&self, doc_name: &str, name_emb: &[f32], doc_desc: &str, desc_emb: &[f32]) -> anyhow::Result<i64> {
        let mut conn = self.get_conn()?;
        use self::schema::schema::documents::dsl::*;

        let new_doc = NewDocument {
            name: doc_name.to_string(),
            name_embedding: Some(Vector::from(name_emb.to_vec())),
            description: Some(doc_desc.to_string()),
            description_embedding: Some(Vector::from(desc_emb.to_vec())),
        };

        let inserted_doc_id: i64 = diesel::insert_into(documents)
            .values(&new_doc)
            .returning(id)
            .get_result(&mut conn)?;

        Ok(inserted_doc_id)
    }

    pub fn list_documents(
        &self,
        limit: Option<i64>,
        offset: Option<i64>,
        keyword: Option<&str>,
        keyword_emb: Option<&[f32]>,
    ) -> anyhow::Result<Vec<DocumentView>> {
        let mut conn = self.get_conn()?;
        use self::schema::schema::documents::dsl::*;

        let limit_value = limit.unwrap_or(5);
        let offset_value = offset.unwrap_or(0);

        if let Some(k) = keyword {
            let trimmed = k.trim();
            if !trimmed.is_empty() {
                if let Some(emb) = keyword_emb {
                    let rrf_limit = (limit_value + offset_value) * 10;
                    let query = r#"
                        WITH vector_search AS (
                            SELECT id, ROW_NUMBER() OVER (
                                ORDER BY (name_embedding <#> $1) + (COALESCE(description_embedding, name_embedding) <#> $1)
                            ) AS vector_rank
                            FROM documents
                            WHERE name_embedding IS NOT NULL OR description_embedding IS NOT NULL
                            LIMIT $4
                        ),
                        keyword_search AS (
                            SELECT id, ROW_NUMBER() OVER (
                                ORDER BY similarity(name, $2) + similarity(COALESCE(description, ''), $2) DESC
                            ) AS keyword_rank
                            FROM documents
                            WHERE name % $2 OR COALESCE(description, '') % $2
                            LIMIT $4
                        )
                        SELECT d.id, d.name, d.description, d.created_at
                        FROM documents d
                        LEFT JOIN vector_search v ON d.id = v.id
                        LEFT JOIN keyword_search k ON d.id = k.id
                        WHERE v.id IS NOT NULL OR k.id IS NOT NULL
                        ORDER BY (COALESCE(1.0 / (60 + v.vector_rank), 0.0) + COALESCE(1.0 / (60 + k.keyword_rank), 0.0)) DESC
                        LIMIT $3 OFFSET $5
                    "#;

                    let rows = sql_query(query)
                        .bind::<pgvector::sql_types::Vector, _>(Vector::from(emb.to_vec()))
                        .bind::<diesel::sql_types::Text, _>(trimmed)
                        .bind::<BigInt, _>(limit_value)
                        .bind::<BigInt, _>(rrf_limit)
                        .bind::<BigInt, _>(offset_value)
                        .load::<DocumentSearchRow>(&mut conn)?;

                    let results = rows
                        .into_iter()
                        .map(|row| DocumentView {
                            id: row.id,
                            name: row.name,
                            description: row.description,
                            created_at: row.created_at,
                        })
                        .collect();
                    return Ok(results);
                }

                let pattern = format!("%{}%", trimmed);
                let mut fallback_query = documents
                    .select(DocumentView::as_select())
                    .filter(
                        name
                            .ilike(pattern.clone())
                            .or(description.is_not_null().and(description.ilike(pattern))),
                    )
                    .into_boxed();

                fallback_query = fallback_query.limit(limit_value).offset(offset_value);
                let results = fallback_query.load::<DocumentView>(&mut conn)?;
                return Ok(results);
            }
        }
        
        let mut query = documents
            .select(DocumentView::as_select())
            .into_boxed();
        
        query = query.limit(limit_value).offset(offset_value);
        
        let results = query.load::<DocumentView>(&mut conn)?;
        Ok(results)
    }

    pub fn delete_document(&self, doc_id: i64) -> anyhow::Result<()> {
        let mut conn = self.get_conn()?;
        use self::schema::schema::documents::dsl::*;
        
        diesel::delete(documents.filter(id.eq(doc_id)))
            .execute(&mut conn)?;
            
        Ok(())
    }

    pub fn insert_memory(
        &self,
        doc_id: i64,
        sum_text: &str,
        sum_emb: Vec<f32>,
        cont_text: &str,
        cont_emb: Vec<f32>,
        meta: Option<serde_json::Value>,
    ) -> anyhow::Result<i64> {
        let mut conn = self.get_conn()?;
        use self::schema::schema::memory_items::dsl::*;

        let new_item = NewMemoryItem {
            document_id: doc_id,
            summary: sum_text.to_string(),
            summary_embedding: Some(Vector::from(sum_emb)),
            content: cont_text.to_string(),
            content_embedding: Some(Vector::from(cont_emb)),
            metadata: meta,
        };

        let inserted_item_id: i64 = diesel::insert_into(memory_items)
            .values(&new_item)
            .returning(id)
            .get_result(&mut conn)?;

        Ok(inserted_item_id)
    }

    pub fn delete_memory(&self, mem_id: i64) -> anyhow::Result<()> {
        let mut conn = self.get_conn()?;
        use self::schema::schema::memory_items::dsl::*;
        
        diesel::delete(memory_items.filter(id.eq(mem_id)))
            .execute(&mut conn)?;
            
        Ok(())
    }

    pub fn search_memory(
        &self,
        doc_id: Option<i64>,
        query_emb: Vec<f32>,
        query_text: &str,
        column: &str,
        limit: i64,
        offset: Option<i64>,
        metadata_filter: Option<serde_json::Value>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        if column != "summary" && column != "content" {
            return Err(anyhow::anyhow!("Invalid column for search"));
        }

        let mut conn = self.get_conn()?;
        let emb_col = format!("{}_embedding", column);
        
        let mut filter_clause = "1=1".to_string();
        if let Some(d_id) = doc_id {
            filter_clause.push_str(&format!(" AND document_id = {}", d_id));
        }

        let mut bind_metadata = false;
        if metadata_filter.is_some() {
            filter_clause.push_str(" AND metadata @> $3");
            bind_metadata = true;
        }

        let limit_param = if bind_metadata { "$4" } else { "$3" };
        let rrf_limit = (limit + offset.unwrap_or(0)) * 10;
        let offset_clause = offset.map(|o| format!(" OFFSET {}", o)).unwrap_or_default();

        let query = format!(
            r#"
            WITH vector_search AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY {} <#> $1) as vector_rank
                FROM memory_items WHERE {}
                ORDER BY {} <#> $1
                LIMIT {}
            ),
            keyword_search AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY similarity({}, $2) DESC) as keyword_rank
                FROM memory_items WHERE {} AND {} % $2
                ORDER BY similarity({}, $2) DESC
                LIMIT {}
            )
            SELECT m.id, m.document_id, m.summary, m.content, m.metadata,
                   (COALESCE(1.0 / (60 + v.vector_rank), 0.0) + COALESCE(1.0 / (60 + k.keyword_rank), 0.0))::float8 as distance
            FROM memory_items m
            LEFT JOIN vector_search v ON m.id = v.id
            LEFT JOIN keyword_search k ON m.id = k.id
            WHERE v.id IS NOT NULL OR k.id IS NOT NULL
            ORDER BY distance DESC
            LIMIT {}{}
            "#,
            emb_col, filter_clause, emb_col, rrf_limit,
            column, filter_clause, column, column, rrf_limit,
            limit_param, offset_clause
        );

        if let Some(meta) = metadata_filter {
            let results = sql_query(query)
                .bind::<pgvector::sql_types::Vector, _>(Vector::from(query_emb))
                .bind::<diesel::sql_types::Text, _>(query_text)
                .bind::<diesel::sql_types::Jsonb, _>(meta)
                .bind::<BigInt, _>(limit)
                .load::<SearchResult>(&mut conn)?;
            Ok(results)
        } else {
            let results = sql_query(query)
                .bind::<pgvector::sql_types::Vector, _>(Vector::from(query_emb))
                .bind::<diesel::sql_types::Text, _>(query_text)
                .bind::<BigInt, _>(limit)
                .load::<SearchResult>(&mut conn)?;
            Ok(results)
        }
    }

    pub fn search_memory_multi(
        &self,
        doc_id: Option<i64>,
        sum_emb: Vec<f32>,
        cont_emb: Vec<f32>,
        query_summary: &str,
        query_content: &str,
        limit: i64,
        offset: Option<i64>,
        metadata_filter: Option<serde_json::Value>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let mut conn = self.get_conn()?;
        
        let mut filter_clause = "1=1".to_string();
        if let Some(d_id) = doc_id {
            filter_clause.push_str(&format!(" AND document_id = {}", d_id));
        }

        let mut bind_metadata = false;
        if metadata_filter.is_some() {
            filter_clause.push_str(" AND metadata @> $5");
            bind_metadata = true;
        }

        let limit_param = if bind_metadata { "$6" } else { "$5" };
        let rrf_limit = (limit + offset.unwrap_or(0)) * 10;
        let offset_clause = offset.map(|o| format!(" OFFSET {}", o)).unwrap_or_default();

        let query = format!(
            r#"
            WITH vector_search AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY (summary_embedding <#> $1) + (content_embedding <#> $2)) as vector_rank
                FROM memory_items WHERE {}
                ORDER BY (summary_embedding <#> $1) + (content_embedding <#> $2)
                LIMIT {}
            ),
            keyword_search AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY similarity(summary, $3) + similarity(content, $4) DESC) as keyword_rank
                FROM memory_items WHERE {} AND (summary % $3 OR content % $4)
                ORDER BY similarity(summary, $3) + similarity(content, $4) DESC
                LIMIT {}
            )
            SELECT m.id, m.document_id, m.summary, m.content, m.metadata,
                   (COALESCE(1.0 / (60 + v.vector_rank), 0.0) + COALESCE(1.0 / (60 + k.keyword_rank), 0.0))::float8 as distance
            FROM memory_items m
            LEFT JOIN vector_search v ON m.id = v.id
            LEFT JOIN keyword_search k ON m.id = k.id
            WHERE v.id IS NOT NULL OR k.id IS NOT NULL
            ORDER BY distance DESC
            LIMIT {}{}
            "#,
            filter_clause, rrf_limit,
            filter_clause, rrf_limit,
            limit_param, offset_clause
        );

        if let Some(meta) = metadata_filter {
            let results = sql_query(query)
                .bind::<pgvector::sql_types::Vector, _>(Vector::from(sum_emb))
                .bind::<pgvector::sql_types::Vector, _>(Vector::from(cont_emb))
                .bind::<diesel::sql_types::Text, _>(query_summary)
                .bind::<diesel::sql_types::Text, _>(query_content)
                .bind::<diesel::sql_types::Jsonb, _>(meta)
                .bind::<BigInt, _>(limit)
                .load::<SearchResult>(&mut conn)?;
            Ok(results)
        } else {
            let results = sql_query(query)
                .bind::<pgvector::sql_types::Vector, _>(Vector::from(sum_emb))
                .bind::<pgvector::sql_types::Vector, _>(Vector::from(cont_emb))
                .bind::<diesel::sql_types::Text, _>(query_summary)
                .bind::<diesel::sql_types::Text, _>(query_content)
                .bind::<BigInt, _>(limit)
                .load::<SearchResult>(&mut conn)?;
            Ok(results)
        }
    }
}
