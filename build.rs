use burn_onnx::{LoadStrategy, ModelGen};

fn main() {
    ModelGen::new()
        .input("assets/model.onnx")
        .out_dir("model/")
        .load_strategy(LoadStrategy::Bytes)
        .run_from_script();
}