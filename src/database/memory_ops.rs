use diesel::prelude::*;
use diesel::sql_query;
use diesel::sql_types::*;
use pgvector::Vector;

use super::models::*;
use super::Database;

impl Database {
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
        use super::schema::schema::memory_items::dsl::*;

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
        use super::schema::schema::memory_items::dsl::*;

        diesel::delete(memory_items.filter(id.eq(mem_id))).execute(&mut conn)?;

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
            emb_col,
            filter_clause,
            emb_col,
            rrf_limit,
            column,
            filter_clause,
            column,
            column,
            rrf_limit,
            limit_param,
            offset_clause
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
            filter_clause,
            rrf_limit,
            filter_clause,
            rrf_limit,
            limit_param,
            offset_clause
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
