pub mod schema {
    diesel::table! {
        use diesel::sql_types::*;

        documents (id) {
            id -> Int8,
            name -> Text,
            description -> Nullable<Text>,
            created_at -> Nullable<Timestamp>,
        }
    }

    diesel::table! {
        use diesel::sql_types::*;
        use pgvector::sql_types::*;

        memory_items (id) {
            id -> Int8,
            document_id -> Int8,
            summary -> Text,
            summary_embedding -> Nullable<Vector>,
            content -> Text,
            content_embedding -> Nullable<Vector>,
            metadata -> Nullable<Jsonb>,
            created_at -> Nullable<Timestamp>,
        }
    }

    diesel::joinable!(memory_items -> documents (document_id));
    diesel::allow_tables_to_appear_in_same_query!(documents, memory_items);
}
