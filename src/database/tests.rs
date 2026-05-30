#[cfg(test)]
mod tests {
    use crate::database::Database;
    use std::env;

    fn get_test_db() -> Option<Database> {
        dotenvy::dotenv().ok();
        let url = env::var("DATABASE_URL").ok()?;
        Database::new(&url).ok()
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
        let tables = db.list_documents(None, None, None, None).unwrap();
        for t in tables {
            if t.name == doc_name {
                let _ = db.delete_document(t.id);
            }
        }

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
            .list_documents(None, None, None, None)
            .expect("Failed to list documents");
        assert!(tables.iter().any(|t| t.id == doc_id));

        assert!(db.delete_document(doc_id).is_ok());

        let tables = db
            .list_documents(None, None, None, None)
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
        let tables = db.list_documents(None, None, None, None).unwrap();
        for t in tables {
            if t.name == doc_name {
                let _ = db.delete_document(t.id);
            }
        }

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
        let docs = db.list_documents(None, None, None, None).unwrap();
        for d in docs {
            if d.name == original_name || d.name == updated_name {
                let _ = db.delete_document(d.id);
            }
        }

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

        let docs = db.list_documents(None, None, None, None).unwrap();
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

        let docs = db.list_documents(None, None, None, None).unwrap();
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
}
