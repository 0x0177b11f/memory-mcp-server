#[cfg(test)]
mod tests {
    use crate::model::helper::init_model;
    use serde_json;

    #[test]
    fn forward_text_returns_non_empty_embedding() {
        let model = init_model(false).expect("init_model should succeed");

        let embedding = model
            .embedding_text("Rust memory MCP test text")
            .expect("embedding_text should succeed");

        assert!(!embedding.is_empty(), "embedding should not be empty");
        assert!(
            embedding.iter().all(|v| v.is_finite()),
            "embedding should contain only finite values"
        );
    }

    #[test]
    fn embedding_text_handle_simple_input() {
        let model = init_model(false).expect("init_model should succeed");

        let embedding = model
            .embedding_text("This is an example sentence")
            .expect("embedding_text should succeed");

        assert!(!embedding.is_empty(), "embedding should not be empty");

        // check size of embedding
        assert_eq!(
            embedding.len(),
            384,
            "embedding length differs from expected: got {}, expected {}",
            embedding.len(),
            384,
        );

        // check embedding values
        let expected_bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/test_embedding.json"
        ));
        let expected: Vec<f32> = serde_json::from_slice(expected_bytes).expect("Failed to parse test embedding JSON");

        for (i, (&value, &expected_val)) in embedding.iter().zip(expected.iter()).enumerate() {
            assert!(
                (value - expected_val).abs() < 1e-4,
                "embedding value at index {i} differs from expected: got {value}, expected {expected_val}"
            );
        }
    }
}
