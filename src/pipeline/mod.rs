mod download;
mod hasher;
mod inference;

pub use download::ImageDownloader;
pub use hasher::PerceptualHasher;
pub use inference::OnnxInference;

use std::sync::Arc;
use std::time::Instant;

use redis::AsyncCommands;
use tracing::{debug, info, instrument, warn};

use crate::config::AppConfig;
use crate::error::{PipelineError, PipelineResult};

/// The fully-assembled processing pipeline.
///
/// Held behind an `Arc` so it can be shared across worker tasks cheaply.
pub struct Pipeline {
    pub downloader: ImageDownloader,
    pub hasher: PerceptualHasher,
    pub inference: OnnxInference,
    pub redis: redis::aio::ConnectionManager,
    pub config: AppConfig,
}

/// The output of a single pipeline run.
#[derive(Debug, serde::Serialize)]
pub struct PipelineOutput {
    /// The original image URL that was processed.
    pub url: String,
    /// Perceptual hash (hex-encoded).
    pub phash: String,
    /// Model confidence score (0.0–1.0).
    pub score: f32,
    /// Whether the score came from the pHash cache.
    pub cache_hit: bool,
    /// Wall-clock processing time in milliseconds.
    pub elapsed_ms: u64,
}

impl Pipeline {
    /// Construct a new pipeline. This loads the ONNX model and connects to Redis.
    pub async fn new(config: &AppConfig) -> PipelineResult<Self> {
        let redis = redis::Client::open(config.redis_uri.as_str())
            .map_err(PipelineError::Redis)?
            .get_connection_manager()
            .await
            .map_err(PipelineError::Redis)?;

        let inference = OnnxInference::new(&config.model_path)?;
        let downloader = ImageDownloader::new(config.max_download_bytes);
        let hasher = PerceptualHasher::new();

        info!(model = %config.model_path, "pipeline initialized");

        Ok(Self {
            downloader,
            hasher,
            inference,
            redis,
            config: config.clone(),
        })
    }

    /// Execute the full pipeline for a single image URL.
    ///
    /// 1. Download → 2. Decode → 3. pHash → 4. Cache check → 5. ONNX → 6. Cache write
    #[instrument(skip(self), fields(url = %url))]
    pub async fn process(self: &Arc<Self>, url: &str) -> PipelineResult<PipelineOutput> {
        let start = Instant::now();

        // ── Step 1 & 2: Download and decode ─────────────────────────────
        let bytes = self.downloader.fetch(url).await?;
        let img = image::load_from_memory(&bytes)?;
        debug!(width = img.width(), height = img.height(), "image decoded");

        // ── Step 3: Compute perceptual hash ─────────────────────────────
        let phash = self.hasher.hash(&img);
        let cache_key = format!("{}{}", self.config.phash_prefix, phash);

        // ── Step 4: Check pHash cache ───────────────────────────────────
        let mut conn = self.redis.clone();
        let cached: Option<f32> = conn.get(&cache_key).await.ok().flatten();

        if let Some(score) = cached {
            debug!(phash = %phash, score, "cache hit — skipping inference");
            return Ok(PipelineOutput {
                url: url.to_owned(),
                phash,
                score,
                cache_hit: true,
                elapsed_ms: start.elapsed().as_millis() as u64,
            });
        }

        // ── Step 5: Run ONNX inference ──────────────────────────────────
        let score = self.inference.predict(&img).await?;

        debug!(phash = %phash, score, "inference complete");

        // ── Step 6: Cache the result ────────────────────────────────────
        let ttl_secs = self.config.phash_cache_ttl.as_secs() as i64;
        let _: () = conn
            .set_ex(&cache_key, score, ttl_secs as u64)
            .await
            .map_err(|e| {
                warn!(error = %e, key = %cache_key, "failed to cache phash score");
                e
            })?;

        Ok(PipelineOutput {
            url: url.to_owned(),
            phash,
            score,
            cache_hit: false,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }
}
