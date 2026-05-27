mod database;
mod model;
mod server;
mod types;

use clap::Parser;
use hyper_util::{
    rt::{TokioExecutor, TokioIo},
    server::conn::auto::Builder,
    service::TowerToHyperService,
};
use rmcp::transport::{StreamableHttpServerConfig, streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager
}};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use server::ServerState;

#[derive(Parser, Debug)]
#[command(name = "memory-mcp-server")]
#[command(about = "Memory MCP server")]
struct Cli {
    /// Enable GPU inference (requires compile-time feature "gpu", defaults to false)
    #[arg(long, default_value_t = false)]
    gpu: bool,

    /// Web server host (defaults to "0.0.0.0")
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Web server port (defaults to 9180)
    #[arg(long, default_value_t = 9180)]
    port: u16,

    /// Maximum number of search results to return (defaults to 20)
    #[arg(long, env = "MAX_SEARCH_RESULTS", default_value_t = 20)]
    max_search_results: i64,

    /// Database URL
    #[arg(long, env = "DATABASE_URL", default_value = "postgres://postgres:password@localhost/memory_mcp_db")]
    db_url: String,

    /// Allowed hosts (comma-separated)
    #[arg(long, env = "ALLOWED_HOSTS", default_value = "localhost,127.0.0.1")]
    allowed_hosts: Option<String>,

    /// Allowed origins (comma-separated)
    #[arg(long, env = "ALLOWED_ORIGINS", default_value = "")]
    allowed_origins: Option<String>,
}

#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    // Load .env file if present
    dotenvy::dotenv().ok();

    // Initialize JSON logger
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    // Parse command line arguments.
    let cli = Cli::parse();
    tracing::info!("Starting memory-mcp-server with args: {:?}", cli);

    // load the model
    let model = Arc::new(Mutex::new(model::helper::init_model(cli.gpu)?));

    // connect to PostgreSQL from environment variables
    tracing::info!("Connecting to database...");
    let db = Arc::new(
        database::Database::new(&cli.db_url)?,
    );
    db.setup_database()?;
    tracing::info!("Database connected and setup completed.");

    let state = ServerState::new(model.clone(), db.clone(), cli.max_search_results);

    let mut config = StreamableHttpServerConfig::default();
    if let Some(hosts_str) = cli.allowed_hosts {
        let hosts = hosts_str.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>();
        tracing::info!("Configuring allowed hosts: {:?}", hosts);
        config = config.with_allowed_hosts(hosts);
    }
    if let Some(origins_str) = cli.allowed_origins {
        let origins = origins_str.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>();
        tracing::info!("Configuring allowed origins: {:?}", origins);
        config = config.with_allowed_origins(origins);
    }

    let service = TowerToHyperService::new(StreamableHttpService::new(
        move || Ok(state.clone()),
        LocalSessionManager::default().into(),
        config,
    ));

    let addr = format!("{}:{}", cli.host, cli.port);
    tracing::info!("Listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    loop {
        let io = tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Received Ctrl-C, shutting down.");
                break;
            },
            accept = listener.accept() => {
                match accept {
                    Ok((stream, remote_addr)) => {
                        tracing::debug!("Accepted connection from {}", remote_addr);
                        TokioIo::new(stream)
                    }
                    Err(e) => {
                        tracing::error!("Failed to accept connection: {}", e);
                        continue;
                    }
                }
            }
        };
        let service = service.clone();
        tokio::spawn(async move {
            if let Err(e) = Builder::new(TokioExecutor::default())
                .serve_connection(io, service)
                .await
            {
                tracing::error!("Error serving connection: {}", e);
            }
        });
    }
    tracing::info!("Server shut down gracefully.");
    Ok(())
}
