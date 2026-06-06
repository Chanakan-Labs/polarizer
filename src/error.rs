use thiserror::Error;

/// Domain-specific errors for the Polarizer pipeline.
///
/// These are _not_ meant to replace `anyhow` for ad-hoc / infrastructure
/// errors — use `anyhow::Error` for those. `PipelineError` captures failures
/// that have distinct recovery semantics (retry vs. skip vs. alert).
#[derive(Debug, Error)]
pub enum PipelineError {
    // ── Download ────────────────────────────────────────────────────────
    #[error("image download failed: {0}")]
    Download(#[from] reqwest::Error),

    #[error("download exceeded maximum size of {max_bytes} bytes")]
    PayloadTooLarge { max_bytes: u64 },

    #[error("upstream returned non-success status {status} for {url}")]
    UpstreamStatus { status: u16, url: String },

    // ── Image processing ────────────────────────────────────────────────
    #[error("image decode failed: {0}")]
    ImageDecode(#[from] image::ImageError),

    // ── Inference ───────────────────────────────────────────────────────
    #[error("ONNX inference failed: {0}")]
    Inference(String),

    // ── Redis / messaging ───────────────────────────────────────────────
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),

    // ── Catch-all ───────────────────────────────────────────────────────
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// Manual `From` impl since `ort::Error` uses a generic parameter and
// thiserror's `#[from]` doesn't work with it.
impl<T: std::fmt::Debug> From<ort::Error<T>> for PipelineError {
    fn from(e: ort::Error<T>) -> Self {
        PipelineError::Inference(format!("{e:?}"))
    }
}

/// Convenience type alias used throughout the pipeline.
pub type PipelineResult<T> = Result<T, PipelineError>;
