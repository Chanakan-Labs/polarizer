use image::DynamicImage;
use ndarray::{Array, Array4, s};
use ort::session::Session;
use tokio::sync::Mutex;
use tracing::debug;

use crate::error::PipelineResult;

/// Expected input dimensions for the ONNX model.
const INPUT_WIDTH: u32 = 224;
const INPUT_HEIGHT: u32 = 224;

/// ImageNet normalization constants (mean and std per RGB channel).
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Wrapper around an `ort::Session` for running ONNX inference.
///
/// The session is loaded once at startup and shared via `Arc<Pipeline>`.
/// Since `ort::Session::run()` requires `&mut self`, the session is
/// protected by a `tokio::sync::Mutex`. Because actual inference is
/// offloaded to `spawn_blocking`, this never blocks the async runtime.
pub struct OnnxInference {
    session: Mutex<Session>,
}

impl OnnxInference {
    /// Load the ONNX model from disk.
    pub fn new(model_path: &str) -> PipelineResult<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("failed to create session builder: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("failed to set intra threads: {e}"))?
            .commit_from_file(model_path)?;

        debug!(
            inputs = ?session.inputs().iter().map(|i| i.name()).collect::<Vec<_>>(),
            outputs = ?session.outputs().iter().map(|o| o.name()).collect::<Vec<_>>(),
            "ONNX model loaded"
        );

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Pre-process image and run inference, returning a confidence score.
    ///
    /// # Pre-processing steps
    ///
    /// 1. Resize to 224×224 using Lanczos3 (high-quality downscale).
    /// 2. Convert to RGB8 pixel buffer.
    /// 3. Normalize each channel: `(pixel / 255.0 - mean) / std`.
    /// 4. Reshape into NCHW format: `[1, 3, 224, 224]`.
    pub async fn predict(&self, img: &DynamicImage) -> PipelineResult<f32> {
        let tensor = Self::preprocess(img);

        // Convert ndarray into an ort Value for the session.
        let input_value = ort::value::Tensor::from_array(tensor)?;

        // Acquire the mutex and run inference.
        let mut session = self.session.lock().await;
        let outputs = session.run(ort::inputs!["input" => input_value])?;

        // Extract the score from the first output tensor.
        // Adjust this indexing based on your specific model's output shape.
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()?;

        // Most classification models output either:
        //   - A single float (score), or
        //   - A [1, N] array of class probabilities.
        // We take the first element as the primary score.
        let score = data.iter().next().copied().unwrap_or(0.0);

        Ok(score)
    }

    /// Convert a `DynamicImage` into an NCHW Float32 tensor with ImageNet normalization.
    fn preprocess(img: &DynamicImage) -> Array4<f32> {
        // 1. Resize
        let resized = img.resize_exact(
            INPUT_WIDTH,
            INPUT_HEIGHT,
            image::imageops::FilterType::Lanczos3,
        );

        // 2. Get raw RGB bytes
        let rgb = resized.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let raw = rgb.into_raw();

        // 3. Build an [H, W, 3] array, normalize, then transpose to [3, H, W].
        let hwc = Array::from_shape_vec((h, w, 3), raw)
            .expect("shape mismatch during tensor construction")
            .mapv(|v| v as f32 / 255.0);

        // Normalize per channel: (val - mean) / std
        let mut nchw = Array4::<f32>::zeros((1, 3, h, w));
        for c in 0..3 {
            let channel = hwc.slice(s![.., .., c]).mapv(|v| (v - MEAN[c]) / STD[c]);
            nchw.slice_mut(s![0, c, .., ..]).assign(&channel);
        }

        nchw
    }
}
