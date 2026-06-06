use std::env;
use std::time::Duration;

use anyhow::{Context, Result};

/// Application configuration sourced entirely from environment variables.
#[derive(Debug, Clone)]
pub struct AppConfig {
    // ── Redis ───────────────────────────────────────────────────────────
    /// Redis connection URI (e.g. `redis://127.0.0.1:6379`).
    pub redis_uri: String,

    /// Name of the Redis stream to consume from.
    pub stream_key: String,

    /// Consumer group name.
    pub consumer_group: String,

    /// Unique consumer name within the group (defaults to hostname).
    pub consumer_name: String,

    /// How many messages to pull per `XREADGROUP` call.
    pub batch_size: usize,

    /// Block timeout for `XREADGROUP`.
    pub block_timeout: Duration,

    // ── Pipeline ────────────────────────────────────────────────────────
    /// Path to the ONNX model file.
    pub model_path: String,

    /// Number of concurrent worker tasks.
    pub worker_count: usize,

    /// Maximum allowed download size in bytes (defense against oversized payloads).
    pub max_download_bytes: u64,

    /// Redis key prefix for pHash cache entries.
    pub phash_prefix: String,

    /// TTL for cached pHash → score mappings.
    pub phash_cache_ttl: Duration,

    /// Redis stream key to push results into.
    pub result_stream_key: String,

    // ── Health ──────────────────────────────────────────────────────────
    /// Port for the health / readiness HTTP server.
    pub health_port: u16,

    // ── Logging ─────────────────────────────────────────────────────────
    /// `tracing` env-filter directive (e.g. `info`, `polarizer=debug`).
    pub log_level: String,
}

impl AppConfig {
    /// Build config from environment variables, applying sensible defaults.
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            // Required
            redis_uri: required("REDIS_URI")?,
            model_path: required("MODEL_PATH")?,

            // Optional with defaults
            stream_key: optional("STREAM_KEY", "polarizer:jobs"),
            consumer_group: optional("CONSUMER_GROUP", "polarizer-workers"),
            consumer_name: optional(
                "CONSUMER_NAME",
                &hostname::get()
                    .map(|h| h.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| "worker-0".into()),
            ),
            batch_size: optional("BATCH_SIZE", "10")
                .parse()
                .context("BATCH_SIZE must be a valid usize")?,
            block_timeout: Duration::from_millis(
                optional("BLOCK_TIMEOUT_MS", "5000")
                    .parse()
                    .context("BLOCK_TIMEOUT_MS must be a valid u64")?,
            ),
            worker_count: optional("WORKER_COUNT", "4")
                .parse()
                .context("WORKER_COUNT must be a valid usize")?,
            max_download_bytes: optional("MAX_DOWNLOAD_BYTES", "20971520") // 20 MiB
                .parse()
                .context("MAX_DOWNLOAD_BYTES must be a valid u64")?,
            phash_prefix: optional("PHASH_PREFIX", "phash:"),
            phash_cache_ttl: Duration::from_secs(
                optional("PHASH_CACHE_TTL_SECS", "604800") // 7 days
                    .parse()
                    .context("PHASH_CACHE_TTL_SECS must be a valid u64")?,
            ),
            result_stream_key: optional("RESULT_STREAM_KEY", "polarizer:results"),
            health_port: optional("HEALTH_PORT", "9090")
                .parse()
                .context("HEALTH_PORT must be a valid u16")?,
            log_level: optional("LOG_LEVEL", "info"),
        })
    }
}

fn required(key: &str) -> Result<String> {
    env::var(key).with_context(|| format!("missing required environment variable: {key}"))
}

fn optional(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_owned())
}
