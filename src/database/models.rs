use super::schema::schema::*;
use diesel::prelude::*;
use pgvector::Vector;
use rmcp::schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub mod stored {
    use super::*;

    #[derive(Insertable, Debug)]
    #[diesel(table_name = documents)]
    pub struct NewDocument {
        pub name: String,
        pub name_embedding: Option<Vector>,
        pub description: Option<String>,
        pub description_embedding: Option<Vector>,
    }

    #[derive(Insertable, Debug)]
    #[diesel(table_name = memory_items)]
    pub struct NewMemoryItem {
        pub document_id: i64,
        pub summary: String,
        pub summary_embedding: Option<Vector>,
        pub content: String,
        pub content_embedding: Option<Vector>,
        pub metadata: Option<serde_json::Value>,
    }
}

pub mod view {
    use super::*;

    #[derive(Queryable, Selectable, Serialize, Deserialize, Debug, JsonSchema)]
    #[diesel(table_name = documents)]
    pub struct DocumentView {
        pub id: i64,
        pub name: String,
        pub description: Option<String>,
        pub created_at: Option<chrono::NaiveDateTime>,
    }

    #[derive(QueryableByName, Debug)]
    pub struct DocumentSearchRow {
        #[diesel(sql_type = diesel::sql_types::BigInt)]
        pub id: i64,
        #[diesel(sql_type = diesel::sql_types::Text)]
        pub name: String,
        #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
        pub description: Option<String>,
        #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Timestamp>)]
        pub created_at: Option<chrono::NaiveDateTime>,
    }
}

pub mod search {
    use super::*;

    #[derive(QueryableByName, Debug, Serialize, JsonSchema)]
    pub struct SearchResult {
        #[diesel(sql_type = diesel::sql_types::BigInt)]
        pub id: i64,
        #[diesel(sql_type = diesel::sql_types::BigInt)]
        pub document_id: i64,
        #[diesel(sql_type = diesel::sql_types::Text)]
        pub summary: String,
        #[diesel(sql_type = diesel::sql_types::Text)]
        pub content: String,
        #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Jsonb>)]
        pub metadata: Option<serde_json::Value>,
        #[diesel(sql_type = diesel::sql_types::Double)]
        pub distance: f64,
    }
}

pub use search::SearchResult;
pub use stored::{NewDocument, NewMemoryItem};
pub(crate) use view::DocumentSearchRow;
pub use view::DocumentView;
