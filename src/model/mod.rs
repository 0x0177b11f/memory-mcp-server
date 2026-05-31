#[allow(clippy::all, clippy::pedantic, clippy::restriction, clippy::nursery)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}

pub mod helper;

#[cfg(test)]
mod tests;
