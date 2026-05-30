mod schema;
mod models;
mod documents_ops;
mod init;
mod memory_ops;
mod migration;

#[cfg(test)]
mod tests;

use diesel::prelude::*;

#[derive(Clone)]
pub struct Database {
    pool: diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>,
}

impl Database {
    pub fn new(database_url: &str) -> anyhow::Result<Self> {
        let manager = diesel::r2d2::ConnectionManager::<PgConnection>::new(database_url);
        let pool = diesel::r2d2::Pool::builder().build(manager)?;

        Ok(Self { pool })
    }

    pub fn get_conn(&self) -> anyhow::Result<diesel::r2d2::PooledConnection<diesel::r2d2::ConnectionManager<PgConnection>>> {
        Ok(self.pool.get()?)
    }
}
