use schemars::JsonSchema;
use serde::Deserialize;

pub mod documents {
    use super::*;

    #[derive(Deserialize, JsonSchema)]
    pub struct CreateDocArgs {
        #[schemars(description = "The name of the document collection")]
        pub name: String,
        #[schemars(description = "A description of the document collection")]
        pub description: String,
    }

    #[derive(Deserialize, JsonSchema)]
    pub struct DeleteDocArgs {
        #[schemars(description = "The ID of the document collection to delete")]
        pub document_id: i64,
    }

    #[derive(Deserialize, JsonSchema)]
    pub struct UpdateDocArgs {
        #[schemars(description = "The ID of the document collection to update")]
        pub document_id: i64,
        #[schemars(description = "The updated name of the document collection")]
        pub name: Option<String>,
        #[schemars(description = "The updated description of the document collection")]
        pub description: Option<String>,
    }

    #[derive(Deserialize, JsonSchema)]
    pub struct ListDocsArgs {
        #[schemars(description = "Optional keyword to search in document name and description")]
        pub keyword: Option<String>,
        #[schemars(description = "The maximum number of results to return")]
        pub limit: Option<i64>,
        #[schemars(description = "The offset for pagination")]
        pub offset: Option<i64>,
    }
}

pub mod memory_items {
    use super::*;

    #[derive(Deserialize, JsonSchema)]
    pub struct InsertMemoryArgs {
        #[schemars(
            description = "The ID of the document collection to insert the memory chunk into"
        )]
        pub document_id: i64,
        #[schemars(description = "A short summary of the memory chunk")]
        pub summary: String,
        #[schemars(description = "The full content of the memory chunk")]
        pub content: String,
        #[schemars(description = "Optional metadata in JSON format for the memory chunk")]
        pub metadata: Option<serde_json::Value>,
    }

    #[derive(Deserialize, JsonSchema)]
    pub struct DeleteMemoryArgs {
        #[schemars(description = "The ID of the memory chunk to delete")]
        pub memory_id: i64,
    }

    #[derive(Deserialize, JsonSchema)]
    pub struct SearchMemoryArgs {
        #[schemars(description = "The ID of the document collection to search in (optional)")]
        pub document_id: Option<i64>,
        #[schemars(description = "The query string to search for")]
        pub query_text: String,
        #[schemars(
            description = "Optional minimum distance threshold; lower-scoring results are filtered out, default is 0.008 for cosine similarity"
        )]
        pub min_distance: Option<f64>,
        #[schemars(description = "The maximum number of results to return")]
        pub limit: Option<i64>,
        #[schemars(description = "The offset for pagination")]
        pub offset: Option<i64>,
        #[schemars(description = "Optional JSON metadata filter (uses @> operator in PostgreSQL)")]
        pub metadata_filter: Option<serde_json::Value>,
    }

    #[derive(Deserialize, JsonSchema)]
    pub struct SearchMemoryMultiArgs {
        #[schemars(description = "The ID of the document collection to search in (optional)")]
        pub document_id: Option<i64>,
        #[schemars(description = "The query string to search for in memory summaries")]
        pub query_summary: String,
        #[schemars(description = "The query string to search for in memory contents")]
        pub query_content: String,
        #[schemars(
            description = "Optional minimum distance threshold; lower-scoring results are filtered out, default is 0.008 for cosine similarity"
        )]
        pub min_distance: Option<f64>,
        #[schemars(description = "The maximum number of results to return")]
        pub limit: Option<i64>,
        #[schemars(description = "The offset for pagination")]
        pub offset: Option<i64>,
        #[schemars(description = "Optional JSON metadata filter (uses @> operator in PostgreSQL)")]
        pub metadata_filter: Option<serde_json::Value>,
    }
}

pub use documents::{CreateDocArgs, DeleteDocArgs, ListDocsArgs, UpdateDocArgs};
pub use memory_items::{
    DeleteMemoryArgs, InsertMemoryArgs, SearchMemoryArgs, SearchMemoryMultiArgs,
};
