use diesel::prelude::*;
use diesel::sql_query;
use diesel::sql_types::*;
use pgvector::Vector;

use super::Database;
use super::RRF_KEYWORD_WEIGHT;
use super::RRF_VECTOR_WEIGHT;
use super::models::*;

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
        doc_id: i64,
        query_emb: Vec<f32>,
        query_text: &str,
        column: &str,
        limit: i64,
        offset: Option<i64>,
        min_distance: f64,
        metadata_filter: Option<serde_json::Value>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        if column != "summary" && column != "content" {
            return Err(anyhow::anyhow!("Invalid column for search"));
        }

        let mut conn = self.get_conn()?;

        let mut materialized_view_clause = format!("document_id = {}", doc_id);

        let mut bind_metadata = false;
        if metadata_filter.is_some() {
            materialized_view_clause.push_str(" AND metadata @> $3");
            bind_metadata = true;
        }

        let limit_param = if bind_metadata { "$4" } else { "$3" };
        let min_distance_param = if bind_metadata { "$5" } else { "$4" };
        let rrf_limit = (limit + offset.unwrap_or(0)) * 10;
        let offset_clause = offset.map(|o| format!(" OFFSET {}", o)).unwrap_or_default();
        
        let emb_col = format!("{}_embedding", column);
        let vector_order_expr = format!("{} <#> $1", emb_col);
        let keyword_order_expr = format!("similarity({}, $2)", column);
        let keyword_where_clause = format!("{} % $2", column);
        let vector_where_clause = format!("{} IS NOT NULL", emb_col);

        let query = format!(
            r#"
            WITH
            scope AS MATERIALIZED (
                SELECT
                    id,
                    summary_embedding,
                    content_embedding,
                    summary,
                    content,
                    document_id,
                    metadata
                FROM memory_items
                WHERE {}
            ),
            vector_search AS (
                SELECT id, ROW_NUMBER() OVER () AS rank
                FROM (
                    SELECT id FROM scope
                    WHERE {}
                    ORDER BY {}
                    LIMIT {}
                ) t
            ),
            keyword_search AS (
                SELECT id, ROW_NUMBER() OVER () AS rank
                FROM (
                    SELECT id FROM scope
                    WHERE {}
                    ORDER BY {} DESC
                    LIMIT {}
                ) t
            ),
             combined_ids AS (
                SELECT id, SUM(weight / (60 + rank))::float8 AS score
                FROM (
                    SELECT id, rank, {}::float8 AS weight FROM vector_search
                    UNION ALL
                    SELECT id, rank, {}::float8 AS weight FROM keyword_search
                ) r
                GROUP BY id
            )
            SELECT
                s.id,
                s.document_id,
                s.summary,
                s.content,
                s.metadata,
                c.score AS score
            FROM combined_ids c
            JOIN scope s ON c.id = s.id
            WHERE c.score >= {}
            ORDER BY c.score DESC
            LIMIT {}{}
            "#,
            materialized_view_clause,
            vector_where_clause,
            vector_order_expr,
            rrf_limit,
            keyword_where_clause,
            keyword_order_expr,
            rrf_limit,
            RRF_VECTOR_WEIGHT,
            RRF_KEYWORD_WEIGHT,
            min_distance_param,
            limit_param,
            offset_clause
        );

        if let Some(meta) = metadata_filter {
            let results = sql_query(query)
                .bind::<pgvector::sql_types::Vector, _>(Vector::from(query_emb))
                .bind::<diesel::sql_types::Text, _>(query_text)
                .bind::<diesel::sql_types::Jsonb, _>(meta)
                .bind::<BigInt, _>(limit)
                .bind::<Double, _>(min_distance)
                .load::<SearchResult>(&mut conn)?;
            Ok(results)
        } else {
            let results = sql_query(query)
                .bind::<pgvector::sql_types::Vector, _>(Vector::from(query_emb))
                .bind::<diesel::sql_types::Text, _>(query_text)
                .bind::<BigInt, _>(limit)
                .bind::<Double, _>(min_distance)
                .load::<SearchResult>(&mut conn)?;
            Ok(results)
        }
    }

    pub fn search_memory_multi(
        &self,
        doc_id: i64,
        sum_emb: Vec<f32>,
        cont_emb: Vec<f32>,
        query_summary: &str,
        query_content: &str,
        limit: i64,
        offset: Option<i64>,
        min_distance: f64,
        metadata_filter: Option<serde_json::Value>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let mut conn = self.get_conn()?;

        let mut id_view_clause = format!("document_id = {}", doc_id);

        let mut bind_metadata = false;
        if metadata_filter.is_some() {
            id_view_clause.push_str(" AND metadata @> $5");
            bind_metadata = true;
        }

        let limit_param = if bind_metadata { "$6" } else { "$5" };
        let min_distance_param = if bind_metadata { "$7" } else { "$6" };
        let rrf_limit = (limit + offset.unwrap_or(0)) * 10;
        let offset_clause = offset.map(|o| format!(" OFFSET {}", o)).unwrap_or_default();

        let query = format!(
            r#"
            WITH
            scope AS MATERIALIZED (
                SELECT
                    id,
                    summary_embedding,
                    content_embedding,
                    summary,
                    content,
                    document_id,
                    metadata
                FROM memory_items
                WHERE {}
            ),
            summary_vector AS (
                SELECT id, ROW_NUMBER() OVER () AS rank
                FROM (
                    SELECT id FROM scope
                    WHERE {}
                        AND summary_embedding IS NOT NULL
                    ORDER BY summary_embedding <#> $1
                    LIMIT {}
                ) t
            ),
            content_vector AS (
                SELECT id, ROW_NUMBER() OVER () AS rank
                FROM (
                    SELECT id FROM scope
                    WHERE {}
                        AND content_embedding IS NOT NULL
                    ORDER BY content_embedding <#> $2
                    LIMIT {}
                ) t
            ),
            summary_keyword AS (
                SELECT id, ROW_NUMBER() OVER () AS rank
                FROM (
                    SELECT id FROM scope
                    WHERE {}
                        AND summary % $3
                    ORDER BY similarity(summary, $3) DESC
                    LIMIT {}
                ) t
            ),
            content_keyword AS (
                SELECT id, ROW_NUMBER() OVER () AS rank
                FROM (
                    SELECT id FROM scope
                    WHERE {}
                        AND content % $4
                    ORDER BY similarity(content, $4) DESC
                    LIMIT {}
                ) t
            ),
            combined_ids AS (
                SELECT id, SUM(weight / (60 + rank))::float8 AS score
                FROM (
                    SELECT id, rank, {}::float8 AS weight FROM summary_vector
                    UNION ALL
                    SELECT id, rank, {}::float8 AS weight FROM content_vector
                    UNION ALL
                    SELECT id, rank, {}::float8 AS weight FROM summary_keyword
                    UNION ALL
                    SELECT id, rank, {}::float8 AS weight FROM content_keyword
                ) r
                GROUP BY id
            )
            SELECT 
                s.id,
                s.document_id,
                s.summary,
                s.content,
                s.metadata,
                c.score AS score
            FROM combined_ids c
            JOIN scope s ON c.id = s.id
            WHERE c.score >= {}
            ORDER BY c.score DESC
            LIMIT {}{}
            "#,
            id_view_clause,
            id_view_clause,
            rrf_limit,
            id_view_clause,
            rrf_limit,
            id_view_clause,
            rrf_limit,
            id_view_clause,
            rrf_limit,
            RRF_VECTOR_WEIGHT,
            RRF_VECTOR_WEIGHT,
            RRF_KEYWORD_WEIGHT,
            RRF_KEYWORD_WEIGHT,
            min_distance_param,
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
                .bind::<Double, _>(min_distance)
                .load::<SearchResult>(&mut conn)?;
            Ok(results)
        } else {
            let results = sql_query(query)
                .bind::<pgvector::sql_types::Vector, _>(Vector::from(sum_emb))
                .bind::<pgvector::sql_types::Vector, _>(Vector::from(cont_emb))
                .bind::<diesel::sql_types::Text, _>(query_summary)
                .bind::<diesel::sql_types::Text, _>(query_content)
                .bind::<BigInt, _>(limit)
                .bind::<Double, _>(min_distance)
                .load::<SearchResult>(&mut conn)?;
            Ok(results)
        }
    }
}
