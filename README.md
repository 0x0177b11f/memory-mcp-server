## Memory MCP Server 

MCP server for providing context preservation to LLMs

## Features

- Uses PostgreSQL as storage backend
- Supports vector search (requires pgvector)
- Uses all-minilm-l6-v2 for embedding vector generation

## Key Dependencies

- [rmcp](https://github.com/modelcontextprotocol/rust-sdk) MCP SDK package
- [Tokio](https://github.com/tokio-rs/tokio) Asynchronous runtime
- [Diesel](https://github.com/diesel-rs/diesel) PostgreSQL client
- [Hyper](https://github.com/hyperium/hyper) HTTP server
- [burn](https://github.com/tracel-ai/burn) Model execution engine

## Build Instructions

You need to place the `model.onnx` file of `all-minilm-l6-v2` into the `./assets` directory, and use the script provided by `burn-onnx` to upgrade the model to `opset version 16`.

If output verification is needed, you need to place the `pytorch_model.bin` of `all-minilm-l6-v2` into the `./assets` directory, and use the `test.py` script to create embedding outputs.

### Build Modes

- Default build is CPU-only (faster build time, suitable for CI/CD and Docker):

```bash
cargo build --release
```

- Enable GPU inference support at compile time:

```bash
cargo build --release --features gpu
```

The runtime `--gpu` flag only takes effect when the binary is built with `--features gpu`.

## Runtime Configuration

The server supports both CLI arguments and environment variables. The `.env` file is loaded automatically at startup.

### Quick Start

```bash
cargo run --release -- \
    --host 0.0.0.0 \
    --port 9180 \
    --max-search-results 20
```

Enable GPU at runtime (only valid when built with `--features gpu`):

```bash
cargo run --release --features gpu -- --gpu
```

### CLI Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--gpu` | bool flag | `false` | Enable GPU inference. Requires compile-time feature `gpu`. |
| `--host` | string | `0.0.0.0` | HTTP server bind host. |
| `--port` | u16 | `9180` | HTTP server bind port. |
| `--max-search-results` | i64 | `20` | Maximum number of search results returned by search tools. |
| `--db-url` | string | `postgres://postgres:password@localhost/memory_mcp_db` | PostgreSQL connection string. |
| `--allowed-hosts` | comma-separated string | `localhost,127.0.0.1` | Allowed hosts for streamable HTTP server. |
| `--allowed-origins` | comma-separated string | empty | Allowed origins for streamable HTTP server. |

### Environment Variables

| Variable | Default | Maps to | Description |
| --- | --- | --- | --- |
| `DATABASE_URL` | `postgres://postgres:password@localhost/memory_mcp_db` | `--db-url` | PostgreSQL connection string. |
| `MAX_SEARCH_RESULTS` | `20` | `--max-search-results` | Max number of returned search results. |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | `--allowed-hosts` | Comma-separated allowed hosts. |
| `ALLOWED_ORIGINS` | empty | `--allowed-origins` | Comma-separated allowed origins. |
| `RUST_LOG` | `info` (fallback in code) | tracing filter | Log level filter (for example `debug`, `info`, `warn`). |

Example `.env`:

```dotenv
DATABASE_URL=postgres://postgres:password@localhost/memory_mcp_db
MAX_SEARCH_RESULTS=20
ALLOWED_HOSTS=localhost,127.0.0.1
ALLOWED_ORIGINS=
RUST_LOG=info
```

Note: when both CLI arguments and environment variables are provided, CLI arguments take precedence.

## Third-Party Model Notice

This project uses model assets from `sentence-transformers/all-MiniLM-L6-v2` on Hugging Face.

- Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- License: Apache License 2.0
- Local license copy: `assets/all-MiniLM-L6-v2-LICENSE.txt`
- Third-party notice details: `THIRD_PARTY_NOTICES.md`

The Docker build process uses the repository model asset (`assets/model.onnx`, managed by Git LFS) during compile time.
The model data is embedded into the compiled binary, so runtime distribution does not require shipping model/tokenizer files under `assets/`.

## Linux Binary Distribution Notice

This project publishes Linux x86_64 static binaries via GitHub Actions CI.

GitHub Releases currently contain a single Linux artifact built for `x86_64-unknown-linux-musl`.

When distributing the Linux release binary (for example via GitHub Releases), distributors should also provide:

- a copy of `LICENSE` for this project;
- third-party attribution details in `THIRD_PARTY_NOTICES.md`;
- the model license copy in `assets/all-MiniLM-L6-v2-LICENSE.txt`.

For this project, Linux release archives do not need to include model files from `assets/` because the model is embedded in the binary.

## MCP Server Features

Designed based on a unified core table structure to replace dynamic table creation which easily leads to Catalog bloat. The system currently contains two core tables:

1. `documents`: Stores metadata of document collections (categories).
2. `memory_items`: Stores specific memory vector chunks.

### Core Table Structure

`documents` table structure:

```sql
CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

`memory_items` table structure:

```sql
CREATE TABLE memory_items (
    id bigserial PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    summary_embedding vector(384),
    content TEXT NOT NULL,
    content_embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Tools provided to AI Agent

| Tool Name | Primary Function | Arguments | Internal Execution Logic |
| --- | --- | --- | --- |
| **`create_document`** | Create a new document collection category | `name` (string) `description` (string) | Registers a new document collection category in the `documents` table and returns the assigned ID. |
| **`list_documents`** | Get a list of all current document collections | None | Queries and returns list information of all `documents`, helping the Agent understand existing categories. |
| **`delete_document`** | Completely delete a document collection and related chunks | `document_id` (integer) | Deletes records from `documents`, utilizing CASCADE deletion to clean up chunk data in `memory_items` simultaneously. |
| **`insert_memory`** | Insert a new vector memory chunk | `document_id` (integer) `summary` (string) `content` (string) | Calls `burn` to convert text into a 384-dimensional vector, inserting it along with the text and foreign key into the `memory_items` table. |
| **`delete_memory`** | **Remove a single specific memory chunk** | `memory_id` (integer) | Precisely deletes the specified `memory_items` record by primary key. Suitable for erasing outdated or incorrect memories. |
| **`search_memory_summary`** | Search memory by summary similarity | `document_id` (integer, optional) `query_text` (string) `limit` (integer, optional) | Vectorizes the query text, performs vector retrieval based on `summary` in `memory_items` (optionally limited to a document collection), and returns Top-K records. |
| **`search_memory_content`** | Search memory by content similarity | `document_id` (integer, optional) `query_text` (string) `limit` (integer, optional) | Vectorizes the query text, performs vector retrieval based on `content` in `memory_items` (optionally limited to a document collection), and returns Top-K records. |
| **`search_memory`** | Search memory by combined summary and content similarity | `document_id` (integer, optional) `query_summary` (string) `query_content` (string) `limit` (integer, optional) | Vectorizes the query texts separately, and simultaneously queries the most relevant memory records with the highest matching degree for `summary` and `content`. |