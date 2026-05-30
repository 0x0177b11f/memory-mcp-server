#[cfg(test)]
mod tests {
    use crate::database::Database;
    use std::env;

    fn get_test_db() -> Option<Database> {
        dotenvy::dotenv().ok();
        let url = env::var("DATABASE_URL").ok()?;
        Database::new(&url).ok()
    }

    fn cleanup_docs_by_exact_name(db: &Database, target_name: &str) {
        let docs = db
            .list_documents(Some(100), None, Some(target_name), None)
            .unwrap_or_default();
        for d in docs {
            if d.name == target_name {
                let _ = db.delete_document(d.id);
            }
        }
    }

    #[test]
    fn test_database_connection() {
        let db = match get_test_db() {
            Some(db) => db,
            None => {
                println!("Skipping test: DATABASE_URL not set");
                return;
            }
        };
        assert!(db.get_conn().is_ok());
    }

    #[test]
    fn test_setup_database() {
        let db = match get_test_db() {
            Some(db) => db,
            None => return,
        };
        assert!(db.setup_database().is_ok());
    }

    #[test]
    fn test_migrate_database() {
        let db = match get_test_db() {
            Some(db) => db,
            None => return,
        };
        assert!(db.migrate_database().is_ok());
    }

    #[test]
    fn test_create_and_delete_document() {
        let db = match get_test_db() {
            Some(db) => db,
            None => return,
        };
        db.setup_database().unwrap();
        let doc_name = "test_doc_collection";

        // Clean up first if exists
        cleanup_docs_by_exact_name(&db, doc_name);

        let name_embedding = vec![0.1; 384];
        let desc_embedding = vec![0.2; 384];
        let doc_id = db
            .create_document(
                doc_name,
                &name_embedding,
                "Test description",
                &desc_embedding,
            )
            .expect("Failed to create document");

        let tables = db
            .list_documents(Some(100), None, Some(doc_name), None)
            .expect("Failed to list documents");
        assert!(tables.iter().any(|t| t.id == doc_id && t.name == doc_name));

        assert!(db.delete_document(doc_id).is_ok());

        let tables = db
            .list_documents(Some(100), None, Some(doc_name), None)
            .expect("Failed to list documents");
        assert!(!tables.iter().any(|t| t.id == doc_id));
    }

    #[test]
    fn test_insert_and_search_memory() {
        let db = match get_test_db() {
            Some(db) => db,
            None => return,
        };
        db.setup_database().unwrap();

        let doc_name = "test_memory_collection";
        cleanup_docs_by_exact_name(&db, doc_name);

        let name_embedding = vec![0.1; 384];
        let desc_embedding = vec![0.2; 384];
        let doc_id = db
            .create_document(doc_name, &name_embedding, "Test", &desc_embedding)
            .unwrap();

        let summary_embedding = vec![0.1; 384];
        let content_embedding = vec![0.2; 384];

        let mem_id = db
            .insert_memory(
                doc_id,
                "Test Summary",
                summary_embedding.clone(),
                "Test Content",
                content_embedding.clone(),
                None,
            )
            .expect("Failed to insert memory");

        let results = db
            .search_memory(
                Some(doc_id),
                summary_embedding,
                "Test Summary",
                "summary",
                1,
                None,
                0.0,
                None,
            )
            .expect("Search failed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].summary, "Test Summary");

        db.delete_memory(mem_id).unwrap();
        db.delete_document(doc_id).unwrap();
    }

    #[test]
    fn test_update_document_name_and_description() {
        let db = match get_test_db() {
            Some(db) => db,
            None => return,
        };
        db.setup_database().unwrap();

        let original_name = "test_update_doc_original";
        let updated_name = "test_update_doc_new_name";
        let original_description = "Original description";
        let updated_description = "Updated description";

        // Best-effort cleanup for deterministic test runs.
        cleanup_docs_by_exact_name(&db, original_name);
        cleanup_docs_by_exact_name(&db, updated_name);

        let name_embedding = vec![0.1; 384];
        let desc_embedding = vec![0.2; 384];
        let doc_id = db
            .create_document(
                original_name,
                &name_embedding,
                original_description,
                &desc_embedding,
            )
            .unwrap();

        let updated_name_embedding = vec![0.3; 384];
        db.update_document(
            doc_id,
            Some(updated_name),
            Some(&updated_name_embedding),
            None,
            None,
        )
        .unwrap();

        let docs = db
            .list_documents(Some(100), None, Some(updated_name), None)
            .unwrap();
        let after_name_update = docs.iter().find(|d| d.id == doc_id).unwrap();
        assert_eq!(after_name_update.name, updated_name);
        assert_eq!(
            after_name_update.description.as_deref(),
            Some(original_description)
        );

        let updated_desc_embedding = vec![0.4; 384];
        db.update_document(
            doc_id,
            None,
            None,
            Some(updated_description),
            Some(&updated_desc_embedding),
        )
        .unwrap();

        let docs = db
            .list_documents(Some(100), None, Some(updated_name), None)
            .unwrap();
        let after_description_update = docs.iter().find(|d| d.id == doc_id).unwrap();
        assert_eq!(after_description_update.name, updated_name);
        assert_eq!(
            after_description_update.description.as_deref(),
            Some(updated_description)
        );

        db.delete_document(doc_id).unwrap();
    }

    #[test]
    fn test_update_document_not_found_returns_error() {
        let db = match get_test_db() {
            Some(db) => db,
            None => return,
        };
        db.setup_database().unwrap();

        let updated_name_embedding = vec![0.3; 384];
        let result = db.update_document(
            -1,
            Some("does-not-exist"),
            Some(&updated_name_embedding),
            None,
            None,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_search_memory_recall_at_k() {
        let db = match get_test_db() {
            Some(db) => db,
            None => return,
        };
        db.setup_database().unwrap();

        let doc_name = "test_recall_collection";
        cleanup_docs_by_exact_name(&db, doc_name);

        let name_embedding = vec![0.1; 384];
        let desc_embedding = vec![0.2; 384];
        let doc_id = db
            .create_document(doc_name, &name_embedding, "Recall test", &desc_embedding)
            .unwrap();

        let query_embedding = vec![1.0; 384];
        let filler_embedding = vec![0.0; 384];

        let relevant_total = 5;
        let k = 5;

        // Relevant memories are crafted to be both keyword- and vector-close to query.
        for idx in 0..relevant_total {
            let summary = format!("recall target memory {}", idx);
            let content = format!("relevant content {}", idx);
            db.insert_memory(
                doc_id,
                &summary,
                query_embedding.clone(),
                &content,
                filler_embedding.clone(),
                None,
            )
            .unwrap();
        }

        // Distractors are vector-far and text-unrelated.
        for idx in 0..25 {
            let summary = format!("noise record {}", idx);
            let content = format!("unrelated content {}", idx);
            db.insert_memory(
                doc_id,
                &summary,
                vec![-1.0; 384],
                &content,
                filler_embedding.clone(),
                None,
            )
            .unwrap();
        }

        let results = db
            .search_memory(
                Some(doc_id),
                query_embedding,
                "recall target memory",
                "summary",
                k,
                None,
                0.0,
                None,
            )
            .unwrap();

        let hit_count = results
            .iter()
            .filter(|r| r.summary.starts_with("recall target memory"))
            .count();
        let recall_at_k = hit_count as f64 / relevant_total as f64;

        assert_eq!(results.len(), k as usize);
        assert!(
            recall_at_k >= 0.8,
            "Recall@{} too low: {:.3}. hits={}, relevant_total={}",
            k,
            recall_at_k,
            hit_count,
            relevant_total
        );

        db.delete_document(doc_id).unwrap();
    }

    #[test]
    fn test_search_memory_multi_recall_at_k() {
        let db = match get_test_db() {
            Some(db) => db,
            None => return,
        };
        db.setup_database().unwrap();

        let doc_name = "test_multi_recall_collection";
        cleanup_docs_by_exact_name(&db, doc_name);

        let name_embedding = vec![0.1; 384];
        let desc_embedding = vec![0.2; 384];
        let doc_id = db
            .create_document(doc_name, &name_embedding, "Multi recall test", &desc_embedding)
            .unwrap();

        let query_sum_embedding = vec![1.0; 384];
        let query_cont_embedding = vec![0.8; 384];

        let relevant_total = 4;
        let k = 4;

        for idx in 0..relevant_total {
            let summary = format!("multi recall summary {}", idx);
            let content = format!("multi recall content {}", idx);
            db.insert_memory(
                doc_id,
                &summary,
                query_sum_embedding.clone(),
                &content,
                query_cont_embedding.clone(),
                None,
            )
            .unwrap();
        }

        for idx in 0..20 {
            let summary = format!("multi noise summary {}", idx);
            let content = format!("multi noise content {}", idx);
            db.insert_memory(
                doc_id,
                &summary,
                vec![-1.0; 384],
                &content,
                vec![-0.8; 384],
                None,
            )
            .unwrap();
        }

        let results = db
            .search_memory_multi(
                Some(doc_id),
                query_sum_embedding,
                query_cont_embedding,
                "multi recall summary",
                "multi recall content",
                k,
                None,
                0.0,
                None,
            )
            .unwrap();

        let hit_count = results
            .iter()
            .filter(|r| r.summary.starts_with("multi recall summary"))
            .count();
        let recall_at_k = hit_count as f64 / relevant_total as f64;

        assert_eq!(results.len(), k as usize);
        assert!(
            recall_at_k >= 0.75,
            "Multi Recall@{} too low: {:.3}. hits={}, relevant_total={}",
            k,
            recall_at_k,
            hit_count,
            relevant_total
        );

        db.delete_document(doc_id).unwrap();
    }

    #[test]
    fn test_search_memory_metadata_filter() {
        let db = match get_test_db() {
            Some(db) => db,
            None => return,
        };
        db.setup_database().unwrap();

        let doc_name = "test_metadata_filter_collection";
        cleanup_docs_by_exact_name(&db, doc_name);

        let name_embedding = vec![0.1; 384];
        let desc_embedding = vec![0.2; 384];
        let doc_id = db
            .create_document(doc_name, &name_embedding, "Metadata filter test", &desc_embedding)
            .unwrap();

        let query_embedding = vec![0.5; 384];

        db.insert_memory(
            doc_id,
            "tenant a memory",
            query_embedding.clone(),
            "content for tenant a",
            query_embedding.clone(),
            Some(serde_json::json!({"tenant": "a", "env": "test"})),
        )
        .unwrap();

        db.insert_memory(
            doc_id,
            "tenant b memory",
            query_embedding.clone(),
            "content for tenant b",
            query_embedding.clone(),
            Some(serde_json::json!({"tenant": "b", "env": "test"})),
        )
        .unwrap();

        let results = db
            .search_memory(
                Some(doc_id),
                query_embedding,
                "tenant",
                "summary",
                10,
                None,
                0.0,
                Some(serde_json::json!({"tenant": "a"})),
            )
            .unwrap();

        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.summary.contains("tenant a")));

        db.delete_document(doc_id).unwrap();
    }

    #[test]
    fn test_search_memory_offset_pagination_no_overlap() {
        let db = match get_test_db() {
            Some(db) => db,
            None => return,
        };
        db.setup_database().unwrap();

        let doc_name = "test_offset_pagination_collection";
        cleanup_docs_by_exact_name(&db, doc_name);

        let name_embedding = vec![0.1; 384];
        let desc_embedding = vec![0.2; 384];
        let doc_id = db
            .create_document(doc_name, &name_embedding, "Offset pagination test", &desc_embedding)
            .unwrap();

        let query_embedding = vec![1.0; 384];

        for idx in 0..8 {
            let summary = format!("page target memory {}", idx);
            let content = format!("page target content {}", idx);
            db.insert_memory(
                doc_id,
                &summary,
                query_embedding.clone(),
                &content,
                vec![0.0; 384],
                None,
            )
            .unwrap();
        }

        let first_page = db
            .search_memory(
                Some(doc_id),
                query_embedding.clone(),
                "page target memory",
                "summary",
                3,
                None,
                0.0,
                None,
            )
            .unwrap();

        let second_page = db
            .search_memory(
                Some(doc_id),
                query_embedding,
                "page target memory",
                "summary",
                3,
                Some(3),
                0.0,
                None,
            )
            .unwrap();

        assert_eq!(first_page.len(), 3);
        assert_eq!(second_page.len(), 3);

        let first_ids: std::collections::HashSet<i64> = first_page.iter().map(|r| r.id).collect();
        let second_ids: std::collections::HashSet<i64> = second_page.iter().map(|r| r.id).collect();

        assert!(first_ids.is_disjoint(&second_ids));

        db.delete_document(doc_id).unwrap();
    }

    #[test]
    fn test_search_memory_min_distance_filters_results() {
        let db = match get_test_db() {
            Some(db) => db,
            None => return,
        };
        db.setup_database().unwrap();

        let doc_name = "test_min_distance_collection";
        cleanup_docs_by_exact_name(&db, doc_name);

        let name_embedding = vec![0.1; 384];
        let desc_embedding = vec![0.2; 384];
        let doc_id = db
            .create_document(doc_name, &name_embedding, "Min distance test", &desc_embedding)
            .unwrap();

        let query_embedding = vec![1.0; 384];
        db.insert_memory(
            doc_id,
            "distance target",
            query_embedding.clone(),
            "distance content",
            query_embedding.clone(),
            None,
        )
        .unwrap();

        // RRF distance is much smaller than 1.0 in this query design, so this threshold must filter all.
        let results = db
            .search_memory(
                Some(doc_id),
                query_embedding,
                "distance target",
                "summary",
                10,
                None,
                1.0,
                None,
            )
            .unwrap();

        assert!(results.is_empty());

        db.delete_document(doc_id).unwrap();
    }

    #[test]
    fn test_search_memory_multi_metadata_filter() {
        let db = match get_test_db() {
            Some(db) => db,
            None => return,
        };
        db.setup_database().unwrap();

        let doc_name = "test_multi_metadata_filter_collection";
        cleanup_docs_by_exact_name(&db, doc_name);

        let name_embedding = vec![0.1; 384];
        let desc_embedding = vec![0.2; 384];
        let doc_id = db
            .create_document(doc_name, &name_embedding, "Multi metadata filter test", &desc_embedding)
            .unwrap();

        let sum_emb = vec![0.6; 384];
        let cont_emb = vec![0.7; 384];

        db.insert_memory(
            doc_id,
            "multi tenant a",
            sum_emb.clone(),
            "multi content a",
            cont_emb.clone(),
            Some(serde_json::json!({"tenant": "a", "scope": "prod"})),
        )
        .unwrap();

        db.insert_memory(
            doc_id,
            "multi tenant b",
            sum_emb.clone(),
            "multi content b",
            cont_emb.clone(),
            Some(serde_json::json!({"tenant": "b", "scope": "prod"})),
        )
        .unwrap();

        let results = db
            .search_memory_multi(
                Some(doc_id),
                sum_emb,
                cont_emb,
                "multi tenant",
                "multi content",
                10,
                None,
                0.0,
                Some(serde_json::json!({"tenant": "a"})),
            )
            .unwrap();

        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.summary.contains("tenant a")));

        db.delete_document(doc_id).unwrap();
    }
}
