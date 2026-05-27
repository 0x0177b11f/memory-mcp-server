use crate::database;
use crate::model;
use crate::types::*;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_handler, tool_router,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, error, debug};

#[derive(Clone)]
pub struct ServerState {
    pub db: Arc<database::Database>,
    pub model: Arc<Mutex<model::helper::RuntimeModel>>,
    pub max_results: i64,
}

#[tool_router]
impl ServerState {
    pub fn new(
        model: Arc<Mutex<model::helper::RuntimeModel>>,
        db: Arc<database::Database>,
        max_results: i64,
    ) -> Self {
        ServerState { db, model, max_results }
    }

    #[tool(
        name = "create_document",
        description = "Create a new document collection"
    )]
    pub async fn create_document(
        &self,
        Parameters(args): Parameters<CreateDocArgs>,
    ) -> Result<CallToolResult, McpError> {
        info!("Creating document: {}", args.name);
        let doc_id = self.db.create_document(&args.name, &args.description)
            .map_err(|e| {
                error!("Failed to create document: {}", e);
                McpError::internal_error(e.to_string(), None)
            })?;

        info!("Document created with ID: {}", doc_id);
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Document collection '{}' created with ID: {}",
            args.name, doc_id
        ))]))
    }

    #[tool(
        name = "list_documents",
        description = "List all document collections and their descriptions"
    )]
    pub async fn list_documents(
        &self,
        Parameters(args): Parameters<ListDocsArgs>,
    ) -> Result<CallToolResult, McpError> {
        info!("Listing documents with limit: {:?}, offset: {:?}", args.limit, args.offset);
        let requested_limit = args.limit.unwrap_or(5);
        let limit = std::cmp::min(requested_limit, self.max_results);
        let docs = self
            .db
            .list_documents(Some(limit), args.offset)
            .map_err(|e| {
                error!("Failed to list documents: {}", e);
                McpError::internal_error(e.to_string(), None)
            })?;
        debug!("Found {} documents", docs.len());
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Document collections: {:?}",
            docs
        ))]))
    }

    #[tool(
        name = "delete_document",
        description = "Delete a document collection and all its memory chunks"
    )]
    pub async fn delete_document(
        &self,
        Parameters(args): Parameters<DeleteDocArgs>,
    ) -> Result<CallToolResult, McpError> {
        info!("Deleting document ID: {}", args.document_id);
        self.db.delete_document(args.document_id)
            .map_err(|e| {
                error!("Failed to delete document: {}", e);
                McpError::internal_error(e.to_string(), None)
            })?;
        info!("Document ID: {} deleted", args.document_id);
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Document collection {} deleted",
            args.document_id
        ))]))
    }

    #[tool(
        name = "insert_memory",
        description = "Insert a memory chunk into a document collection"
    )]
    pub async fn insert_memory(
        &self,
        Parameters(args): Parameters<InsertMemoryArgs>,
    ) -> Result<CallToolResult, McpError> {
        info!("Inserting memory into document ID: {}", args.document_id);
        let sum_emb = self
            .model
            .lock()
            .await
            .embedding_text(&args.summary)
            .map_err(|e| {
                error!("Failed to embed summary: {}", e);
                McpError::internal_error(e, None)
            })?;
        let cont_emb = self
            .model
            .lock()
            .await
            .embedding_text(&args.content)
            .map_err(|e| {
                error!("Failed to embed content: {}", e);
                McpError::internal_error(e, None)
            })?;
        self.db
            .insert_memory(
                args.document_id,
                &args.summary,
                sum_emb,
                &args.content,
                cont_emb,
                args.metadata.clone(),
            )
            .map_err(|e| {
                error!("Failed to insert memory: {}", e);
                McpError::internal_error(e.to_string(), None)
            })?;
        info!("Memory successfully inserted into document ID: {}", args.document_id);
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Memory chunk inserted into document {}",
            args.document_id
        ))]))
    }

    #[tool(
        name = "delete_memory",
        description = "Delete a specific memory chunk"
    )]
    pub async fn delete_memory(
        &self,
        Parameters(args): Parameters<DeleteMemoryArgs>,
    ) -> Result<CallToolResult, McpError> {
        info!("Deleting memory ID: {}", args.memory_id);
        self.db
            .delete_memory(args.memory_id)
            .map_err(|e| {
                error!("Failed to delete memory: {}", e);
                McpError::internal_error(e.to_string(), None)
            })?;
        info!("Memory ID: {} deleted", args.memory_id);
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Memory chunk {} deleted",
            args.memory_id
        ))]))
    }

    #[tool(
        name = "search_memory_summary",
        description = "Search memory chunks by summary similarity"
    )]
    pub async fn search_memory_summary(
        &self,
        Parameters(args): Parameters<SearchMemoryArgs>,
    ) -> Result<CallToolResult, McpError> {
        info!("Searching memory by summary, doc_id: {:?}, query: '{}'", args.document_id, args.query_text);
        let query_emb = self
            .model
            .lock()
            .await
            .embedding_text(&args.query_text)
            .map_err(|e| {
                error!("Failed to embed query: {}", e);
                McpError::internal_error(e, None)
            })?;
        let requested_limit = args.limit.unwrap_or(5);
        let limit = std::cmp::min(requested_limit, self.max_results);
        let results = self
            .db
            .search_memory(
                args.document_id,
                query_emb,
                &args.query_text,
                "summary",
                limit,
                args.offset,
                args.metadata_filter.clone(),
            )
            .map_err(|e| {
                error!("Failed to search memory by summary: {}", e);
                McpError::internal_error(e.to_string(), None)
            })?;

        debug!("Search memory summary returned {} results", results.len());
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Search results: {:?}",
            results
        ))]))
    }

    #[tool(
        name = "search_memory_content",
        description = "Search memory chunks by content similarity"
    )]
    pub async fn search_memory_content(
        &self,
        Parameters(args): Parameters<SearchMemoryArgs>,
    ) -> Result<CallToolResult, McpError> {
        info!("Searching memory by content, doc_id: {:?}, query: '{}'", args.document_id, args.query_text);
        let query_emb = self
            .model
            .lock()
            .await
            .embedding_text(&args.query_text)
            .map_err(|e| {
                error!("Failed to embed query: {}", e);
                McpError::internal_error(e, None)
            })?;
        let requested_limit = args.limit.unwrap_or(5);
        let limit = std::cmp::min(requested_limit, self.max_results);
        let results = self
            .db
            .search_memory(
                args.document_id,
                query_emb,
                &args.query_text,
                "content",
                limit,
                args.offset,
                args.metadata_filter.clone(),
            )
            .map_err(|e| {
                error!("Failed to search memory by content: {}", e);
                McpError::internal_error(e.to_string(), None)
            })?;
        debug!("Search memory content returned {} results", results.len());
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Search results: {:?}",
            results
        ))]))
    }

    #[tool(
        name = "search_memory",
        description = "Search memory chunks by summary and content similarity"
    )]
    pub async fn search_memory(
        &self,
        Parameters(args): Parameters<SearchMemoryMultiArgs>,
    ) -> Result<CallToolResult, McpError> {
        info!("Searching memory, doc_id: {:?}, summary query: '{}', content query: '{}'", args.document_id, args.query_summary, args.query_content);
        let sum_emb = self
            .model
            .lock()
            .await
            .embedding_text(&args.query_summary)
            .map_err(|e| {
                error!("Failed to embed summary query: {}", e);
                McpError::internal_error(e, None)
            })?;
        let cont_emb = self
            .model
            .lock()
            .await
            .embedding_text(&args.query_content)
            .map_err(|e| {
                error!("Failed to embed content query: {}", e);
                McpError::internal_error(e, None)
            })?;

        let requested_limit = args.limit.unwrap_or(5);
        let limit = std::cmp::min(requested_limit, self.max_results);
        let results = self
            .db
            .search_memory_multi(
                args.document_id,
                sum_emb,
                cont_emb,
                &args.query_summary,
                &args.query_content,
                limit,
                args.offset,
                args.metadata_filter.clone(),
            )
            .map_err(|e| {
                error!("Failed to search memory: {}", e);
                McpError::internal_error(e.to_string(), None)
            })?;

        debug!("Search memory returned {} results", results.len());
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Search results: {:?}",
            results
        ))]))
    }
}

#[tool_handler(
    name = "memory-mcp-server",
    version = "0.1.0",
    instructions = "A memory MCP server with vector search"
)]
impl ServerHandler for ServerState {}
