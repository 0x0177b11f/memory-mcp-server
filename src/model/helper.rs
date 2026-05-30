use crate::model::generated::Model;
use burn::backend::flex::{Flex, FlexDevice};
#[cfg(feature = "gpu")]
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::backend::Backend;
use burn::tensor::{Bytes, Int, Tensor, TensorData};
use tokenizers::Tokenizer;
#[cfg(not(feature = "gpu"))]
use tracing::warn;
use tracing::{debug, error, info};

static WEIGHT_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model/model.bpk"));
static TOKENIZER_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/tokenizer.json"
));

pub enum RuntimeDevice {
    #[cfg(feature = "gpu")]
    Gpu(WgpuDevice),
    Cpu(FlexDevice),
}

pub enum RuntimeModel {
    #[cfg(feature = "gpu")]
    Gpu {
        device: WgpuDevice,
        model: Model<Wgpu>,
        tokenizer: Box<Tokenizer>,
    },
    Cpu {
        device: FlexDevice,
        model: Model<Flex>,
        tokenizer: Box<Tokenizer>,
    },
}

impl RuntimeModel {
    fn mean_pooling<B: Backend>(
        &self,
        model_output: Tensor<B, 3>,
        attention_mask: Tensor<B, 2, burn::prelude::Int>,
    ) -> Tensor<B, 2> {
        let token_embeddings = model_output;
        let input_mask_expanded: Tensor<B, 3> = attention_mask.float().unsqueeze_dim(2);

        let sum_embeddings = token_embeddings.mul(input_mask_expanded.clone()).sum_dim(1); // Result dimension: [batch_size, 1, hidden_size]

        let sum_mask = input_mask_expanded
            .sum_dim(1) // [batch_size, 1, 1]
            .clamp_min(1e-9);

        sum_embeddings.div(sum_mask).squeeze_dim(1)
    }

    fn normalize<B: Backend>(&self, sentence_embeddings: Tensor<B, 2>) -> Tensor<B, 2> {
        let square_sum = sentence_embeddings.clone().powf_scalar(2.0).sum_dim(1);
        let l2_norm = square_sum.sqrt().clamp_min(1e-9);
        sentence_embeddings.div(l2_norm)
    }

    fn tokenizer(&self) -> &Tokenizer {
        match self {
            #[cfg(feature = "gpu")]
            RuntimeModel::Gpu { tokenizer, .. } => tokenizer,
            RuntimeModel::Cpu { tokenizer, .. } => tokenizer,
        }
    }

    fn run_embedding<B: Backend>(
        &self,
        device: &B::Device,
        model: &Model<B>,
        input_ids: TensorData,
        attention_mask: TensorData,
        token_type_ids: TensorData,
    ) -> TensorData {
        let input_ids = Tensor::<B, 2, Int>::from_data(input_ids, device);
        let attention_mask = Tensor::<B, 2, Int>::from_data(attention_mask, device);
        let token_type_ids = Tensor::<B, 2, Int>::from_data(token_type_ids, device);

        let output = model.forward(input_ids, attention_mask.clone(), token_type_ids);

        self.normalize(self.mean_pooling(output, attention_mask))
            .to_data()
    }

    pub fn embedding(
        &self,
        input_ids: TensorData,
        attention_mask: TensorData,
        token_type_ids: TensorData,
    ) -> TensorData {
        match self {
            #[cfg(feature = "gpu")]
            RuntimeModel::Gpu {
                device,
                model,
                tokenizer: _,
            } => {
                self.run_embedding::<Wgpu>(device, model, input_ids, attention_mask, token_type_ids)
            }
            RuntimeModel::Cpu {
                device,
                model,
                tokenizer: _,
            } => {
                self.run_embedding::<Flex>(device, model, input_ids, attention_mask, token_type_ids)
            }
        }
    }

    pub fn embedding_text(&self, text: &str) -> Result<Vec<f32>, String> {
        debug!("Embedding text with length: {}", text.len());
        let encoding = self.tokenizer().encode(text, true).map_err(|e| {
            error!("Failed to tokenize input text: {}", e);
            format!("failed to tokenize input text: {e}")
        })?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| i64::from(m))
            .collect();
        let token_type_ids: Vec<i64> = encoding
            .get_type_ids()
            .iter()
            .map(|&id| i64::from(id))
            .collect();

        let seq_len = input_ids.len();
        if seq_len == 0 {
            error!("Tokenizer produced empty input");
            return Err("tokenizer produced empty input".to_string());
        }

        let output = self.embedding(
            TensorData::new(input_ids, [1, seq_len]),
            TensorData::new(attention_mask.clone(), [1, seq_len]),
            TensorData::new(token_type_ids, [1, seq_len]),
        );

        let values = output.to_vec::<f32>().map_err(|e| {
            error!("Failed to read model output: {}", e);
            format!("failed to read model output: {e}")
        })?;

        if values.is_empty() {
            error!("Model output is empty");
            return Err("model output is empty".to_string());
        }

        Ok(values)
    }
}

#[cfg(feature = "gpu")]
fn select_device(enable_gpu: bool) -> RuntimeDevice {
    if enable_gpu {
        info!("Selecting GPU device for model inference");
        RuntimeDevice::Gpu(WgpuDevice::default())
    } else {
        info!("Selecting CPU (Flex) device for model inference");
        RuntimeDevice::Cpu(FlexDevice::default())
    }
}

#[cfg(not(feature = "gpu"))]
fn select_device(enable_gpu: bool) -> RuntimeDevice {
    if enable_gpu {
        warn!(
            "--gpu requested, but binary was built without the 'gpu' feature; falling back to CPU"
        );
    }
    info!("Selecting CPU (Flex) device for model inference");
    RuntimeDevice::Cpu(FlexDevice::default())
}

pub fn init_model(enable_gpu: bool) -> anyhow::Result<RuntimeModel> {
    info!("Initializing model");
    let weight_bytes = Bytes::from_shared(
        WEIGHT_BYTES.into(),
        burn::tensor::AllocationProperty::Native,
    );

    let device = select_device(enable_gpu);
    info!("Loading tokenizer");
    let tokenizer = Tokenizer::from_bytes(TOKENIZER_BYTES).map_err(|e| {
        error!("Failed to load tokenizer: {}", e);
        anyhow::anyhow!("failed to load tokenizer: {e}")
    })?;

    info!("Instantiating runtime model");
    let model = match &device {
        #[cfg(feature = "gpu")]
        RuntimeDevice::Gpu(d) => RuntimeModel::Gpu {
            device: d.clone(),
            model: Model::from_bytes(weight_bytes.clone(), d),
            tokenizer: Box::new(tokenizer),
        },
        RuntimeDevice::Cpu(d) => RuntimeModel::Cpu {
            device: d.clone(),
            model: Model::from_bytes(weight_bytes, d),
            tokenizer: Box::new(tokenizer),
        },
    };

    info!("Model initialized successfully");
    Ok(model)
}
