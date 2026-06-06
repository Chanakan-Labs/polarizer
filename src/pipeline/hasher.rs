use image::DynamicImage;
use image_hasher::{HasherConfig, HashAlg};

/// Perceptual hasher using DCT-based pHash.
///
/// The hash is deterministic and produces a compact hex string that can be
/// compared with Hamming distance for near-duplicate detection.
pub struct PerceptualHasher {
    config: HasherConfig,
}

impl PerceptualHasher {
    pub fn new() -> Self {
        let config = HasherConfig::new()
            .hash_alg(HashAlg::DoubleGradient)
            .hash_size(16, 16);

        Self { config }
    }

    /// Compute the perceptual hash and return it as a hex-encoded string.
    pub fn hash(&self, img: &DynamicImage) -> String {
        let hasher = self.config.to_hasher();
        let hash = hasher.hash_image(img);
        hash.to_base64()
    }
}
