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
    fn test_create_and_delete_document() {
        let db = match get_test_db() {
            Some(db) => db,
            None => return,
        };
        db.setup_database().unwrap();
        let doc_name = "test_doc_collection";
        
        // Clean up first if exists
        let tables = db.list_documents(None, None).unwrap();
        for t in tables {
            if t.name == doc_name {
                let _ = db.delete_document(t.id);
            }
        }

        let doc_id = db.create_document(doc_name, "Test description").expect("Failed to create document");
        
        let tables = db.list_documents(None, None).expect("Failed to list documents");
        assert!(tables.iter().any(|t| t.id == doc_id));

        assert!(db.delete_document(doc_id).is_ok());
        
        let tables = db.list_documents(None, None).expect("Failed to list documents");
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
        let tables = db.list_documents(None, None).unwrap();
        for t in tables {
            if t.name == doc_name {
                let _ = db.delete_document(t.id);
            }
        }
        
        let doc_id = db.create_document(doc_name, "Test").unwrap();

        let summary_embedding = vec![0.1; 384];
        let content_embedding = vec![0.2; 384];

        let mem_id = db.insert_memory(
            doc_id,
            "Test Summary",
            summary_embedding.clone(),
            "Test Content",
            content_embedding.clone(),
            None,
        ).expect("Failed to insert memory");

        let results = db.search_memory(Some(doc_id), summary_embedding, "Test Summary", "summary", 1, None, None)
            .expect("Search failed");
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].summary, "Test Summary");

        db.delete_memory(mem_id).unwrap();
        db.delete_document(doc_id).unwrap();
    }
}
