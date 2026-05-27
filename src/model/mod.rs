mod generated {
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}

pub mod helper;

#[cfg(test)]
mod tests;
