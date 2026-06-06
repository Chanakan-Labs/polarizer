use image::DynamicImage;
use ndarray::{Array, Array4, s};
use ort::session::Session;
use tracing::debug;

use crate::config::AppConfig;
use crate::error::PipelineResult;

/// Configuration for how images are pre-processed before inference.
/// All values are sourced from `AppConfig` at construction time.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Square image dimension the model expects (e.g. 384 for ViT-384).
    pub image_size: u32,
    /// Per-channel mean for normalization.
    pub mean: [f32; 3],
    /// Per-channel standard deviation for normalization.
    pub std: [f32; 3],
}

/// Result of a single inference pass.
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    /// Probability for the target label after softmax (0.0–1.0).
    pub score: f32,
    /// Human-readable label name (e.g. "nsfw").
    pub label: String,
    /// Raw logits from the model (all classes).
    pub logits: Vec<f32>,
}

/// Wrapper around an `ort::Session` for running ONNX inference.
///
/// The session is loaded once at startup and shared via `Arc<Pipeline>`.
/// Since `ort::Session::run()` requires `&mut self`, the session is
/// protected by a `tokio::sync::Mutex`.
pub struct OnnxInference {
    session: std::sync::Arc<std::sync::Mutex<Session>>,
    preprocess: PreprocessConfig,
    input_name: String,
    labels: Vec<String>,
}

impl OnnxInference {
    /// Load the ONNX model from disk using settings from `AppConfig`.
    pub fn new(config: &AppConfig) -> PipelineResult<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("failed to create session builder: {e}"))?
            .with_intra_threads(config.model_intra_threads)
            .map_err(|e| anyhow::anyhow!("failed to set intra threads: {e}"))?
            .commit_from_file(&config.model_path)?;

        debug!(
            inputs = ?session.inputs().iter().map(|i| i.name()).collect::<Vec<_>>(),
            outputs = ?session.outputs().iter().map(|o| o.name()).collect::<Vec<_>>(),
            image_size = config.model_image_size,
            input_name = %config.model_input_name,
            labels = ?config.model_labels,
            "ONNX model loaded"
        );

        Ok(Self {
            session: std::sync::Arc::new(std::sync::Mutex::new(session)),
            preprocess: PreprocessConfig {
                image_size: config.model_image_size,
                mean: config.model_image_mean,
                std: config.model_image_std,
            },
            input_name: config.model_input_name.clone(),
            labels: config.model_labels.clone(),
        })
    }

    /// Pre-process image and run inference, returning a structured result.
    pub async fn predict(&self, img: &DynamicImage) -> PipelineResult<InferenceOutput> {
        let tensor = self.preprocess(img);

        let input_value = ort::value::Tensor::from_array(tensor)?;
        let session = std::sync::Arc::clone(&self.session);
        let input_name = self.input_name.clone();

        // Run CPU-heavy inference on Tokio's blocking thread pool to avoid stalling the async runtime.
        let logits: Vec<f32> = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f32>> {
            let mut session = session.lock().unwrap();
            let outputs = session.run(ort::inputs![input_name.as_str() => input_value])?;

            // Extract raw logits from the first output tensor inside the closure
            // because outputs holds a lifetime bound to the session lock.
            let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
            Ok(data.to_vec())
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking failed: {e}"))??;

        // Apply softmax to convert logits → probabilities.
        let probabilities = softmax(&logits);

        // Extract the label with the highest probability.
        let (max_index, max_score) = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        let score = *max_score;
        let label = self
            .labels
            .get(max_index)
            .cloned()
            .unwrap_or_else(|| format!("class_{}", max_index));

        Ok(InferenceOutput {
            score,
            label,
            logits,
        })
    }

    /// Convert a `DynamicImage` into an NCHW Float32 tensor with configurable normalization.
    fn preprocess(&self, img: &DynamicImage) -> Array4<f32> {
        let size = self.preprocess.image_size;
        let mean = self.preprocess.mean;
        let std = self.preprocess.std;

        // 1. Resize to the exact dimensions the model expects.
        let resized = img.resize_exact(size, size, image::imageops::FilterType::Lanczos3);

        // 2. Get raw RGB bytes.
        let rgb = resized.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let raw = rgb.into_raw();

        // 3. Build [H, W, 3] array, scale to [0, 1], then normalize per-channel.
        let hwc = Array::from_shape_vec((h, w, 3), raw)
            .expect("shape mismatch during tensor construction")
            .mapv(|v| v as f32 / 255.0);

        // 4. Transpose HWC → NCHW and apply (val - mean) / std per channel.
        let mut nchw = Array4::<f32>::zeros((1, 3, h, w));
        for c in 0..3 {
            let channel = hwc.slice(s![.., .., c]).mapv(|v| (v - mean[c]) / std[c]);
            nchw.slice_mut(s![0, c, .., ..]).assign(&channel);
        }

        nchw
    }
}

/// Numerically stable softmax over a slice of logits.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}
