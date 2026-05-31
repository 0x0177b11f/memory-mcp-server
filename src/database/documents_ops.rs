use diesel::prelude::*;
use diesel::sql_query;
use diesel::sql_types::*;
use pgvector::Vector;

use super::Database;
use super::RRF_KEYWORD_WEIGHT;
use super::RRF_VECTOR_WEIGHT;
use super::models::*;

impl Database {
    pub fn create_document(
        &self,
        doc_name: &str,
        name_emb: &[f32],
        doc_desc: &str,
        desc_emb: &[f32],
    ) -> anyhow::Result<i64> {
        let mut conn = self.get_conn()?;
        use super::schema::schema::documents::dsl::*;

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
        use super::schema::schema::documents::dsl::*;

        let limit_value = limit.unwrap_or(5);
        let offset_value = offset.unwrap_or(0);

        if let Some(k) = keyword {
            let trimmed = k.trim();
            if !trimmed.is_empty() {
                if let Some(emb) = keyword_emb {
                    let rrf_limit = (limit_value + offset_value) * 10;
                    let query = format!(
                        r#"
                        WITH
                        name_vector AS (
                            SELECT id, ROW_NUMBER() OVER () AS rank
                            FROM (
                                SELECT id FROM documents
                                WHERE name_embedding IS NOT NULL
                                ORDER BY name_embedding <#> $1
                                LIMIT $4
                            ) t
                        ),
                        description_vector AS (
                            SELECT id, ROW_NUMBER() OVER () AS rank
                            FROM (
                                SELECT id FROM documents
                                WHERE description_embedding IS NOT NULL
                                ORDER BY description_embedding <#> $1
                                LIMIT $4
                            ) t
                        ),
                        name_keyword AS (
                            SELECT id, ROW_NUMBER() OVER () AS rank
                            FROM (
                                SELECT id FROM documents
                                WHERE name % $2
                                ORDER BY similarity(name, $2) DESC
                                LIMIT $4
                            ) t
                        ),
                        description_keyword AS (
                            SELECT id, ROW_NUMBER() OVER () AS rank
                            FROM (
                                SELECT id FROM documents
                                WHERE COALESCE(description, '') % $2
                                ORDER BY similarity(COALESCE(description, ''), $2) DESC
                                LIMIT $4
                            ) t
                        ),
                        combined_ids AS (
                            SELECT id, SUM(weight / (60 + rank))::float8 AS score
                            FROM (
                                SELECT id, rank, {}::float8 AS weight FROM name_vector
                                UNION ALL
                                SELECT id, rank, {}::float8 AS weight FROM description_vector
                                UNION ALL
                                SELECT id, rank, {}::float8 AS weight FROM name_keyword
                                UNION ALL
                                SELECT id, rank, {}::float8 AS weight FROM description_keyword
                            ) r
                            GROUP BY id
                        )
                        SELECT
                            s.id,
                            s.name,
                            s.description,
                            s.created_at
                        FROM combined_ids c
                        JOIN documents s ON c.id = s.id
                        ORDER BY c.score DESC
                        LIMIT $3 OFFSET $5
                        "#,
                        RRF_VECTOR_WEIGHT,
                        RRF_VECTOR_WEIGHT,
                        RRF_KEYWORD_WEIGHT,
                        RRF_KEYWORD_WEIGHT
                    );

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
                let fallback_query = documents
                    .select(DocumentView::as_select())
                    .filter(
                        name.ilike(pattern.clone())
                            .or(description.is_not_null().and(description.ilike(pattern))),
                    )
                    .into_boxed()
                    .limit(limit_value)
                    .offset(offset_value);

                let results = fallback_query.load::<DocumentView>(&mut conn)?;
                return Ok(results);
            }
        }

        let query = documents
            .select(DocumentView::as_select())
            .into_boxed()
            .limit(limit_value)
            .offset(offset_value);
        let results = query.load::<DocumentView>(&mut conn)?;
        Ok(results)
    }

    pub fn delete_document(&self, doc_id: i64) -> anyhow::Result<()> {
        let mut conn = self.get_conn()?;
        use super::schema::schema::documents::dsl::*;

        diesel::delete(documents.filter(id.eq(doc_id))).execute(&mut conn)?;

        Ok(())
    }

    pub fn update_document(
        &self,
        doc_id: i64,
        doc_name: Option<&str>,
        name_emb: Option<&[f32]>,
        doc_desc: Option<&str>,
        desc_emb: Option<&[f32]>,
    ) -> anyhow::Result<()> {
        let mut conn = self.get_conn()?;

        let query = r#"
            UPDATE documents
            SET
                name = COALESCE($2, name),
                name_embedding = COALESCE($3, name_embedding),
                description = COALESCE($4, description),
                description_embedding = COALESCE($5, description_embedding)
            WHERE id = $1
        "#;

        let name_emb_vector = name_emb.map(|v| Vector::from(v.to_vec()));
        let desc_emb_vector = desc_emb.map(|v| Vector::from(v.to_vec()));

        let updated_rows = sql_query(query)
            .bind::<BigInt, _>(doc_id)
            .bind::<Nullable<Text>, _>(doc_name)
            .bind::<Nullable<pgvector::sql_types::Vector>, _>(name_emb_vector)
            .bind::<Nullable<Text>, _>(doc_desc)
            .bind::<Nullable<pgvector::sql_types::Vector>, _>(desc_emb_vector)
            .execute(&mut conn)?;

        if updated_rows == 0 {
            return Err(anyhow::anyhow!("Document collection {} not found", doc_id));
        }

        Ok(())
    }
}
