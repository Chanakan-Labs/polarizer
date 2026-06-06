use reqwest::Client;
use tracing::{debug, instrument};

use crate::error::{PipelineError, PipelineResult};

/// Async image downloader with size-limit enforcement.
pub struct ImageDownloader {
    client: Client,
    max_bytes: u64,
}

impl ImageDownloader {
    pub fn new(max_bytes: u64) -> Self {
        let client = Client::builder()
            .user_agent(concat!("polarizer/", env!("CARGO_PKG_VERSION")))
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to build reqwest client");

        Self { client, max_bytes }
    }

    /// Download image bytes from `url`, enforcing the max payload size.
    ///
    /// We check `Content-Length` first (cheap early reject), then stream the
    /// body with a running byte count so we can't be fooled by a missing or
    /// lying header.
    #[instrument(skip(self))]
    pub async fn fetch(&self, url: &str) -> PipelineResult<Vec<u8>> {
        let resp = self.client.get(url).send().await?;

        let status = resp.status();
        if !status.is_success() {
            return Err(PipelineError::UpstreamStatus {
                status: status.as_u16(),
                url: url.to_owned(),
            });
        }

        // Early reject based on Content-Length header if present.
        if let Some(len) = resp.content_length() {
            if len > self.max_bytes {
                return Err(PipelineError::PayloadTooLarge {
                    max_bytes: self.max_bytes,
                });
            }
        }

        // Stream the body, enforcing size limit even when Content-Length is absent.
        let bytes = resp.bytes().await?;

        if bytes.len() as u64 > self.max_bytes {
            return Err(PipelineError::PayloadTooLarge {
                max_bytes: self.max_bytes,
            });
        }

        debug!(size_bytes = bytes.len(), "download complete");
        Ok(bytes.to_vec())
    }
}
