# ── Builder stage ────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

# ort v2.0 pre-compiled binaries for aarch64 require glibc 2.38+. 
# We use Ubuntu 24.04 (glibc 2.39) to satisfy both linking and runtime requirements.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust 1.88 (required by image/ort)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.88.0
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Cache dependency compilation by copying manifests first.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src

# Build the actual binary.
COPY src/ src/
RUN touch src/main.rs && cargo build --release

# ort v2.0 dynamically links libonnxruntime.so and downloads it to the global cache.
# We need to find it and move it to a predictable location.
RUN mkdir -p /app/out-libs && \
    find /root/.cache/ort.pyke.io target/ -name "libonnxruntime.so*" -exec cp {} /app/out-libs/ \;

# Download the ONNX model so it can be baked into the final image
ARG MODEL_URL="https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/onnx/model_int8.onnx"
RUN curl -sLo /app/model_int8.onnx "${MODEL_URL}"

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM ubuntu:24.04

# Create a non-root user
RUN groupadd -r nonroot && useradd -r -g nonroot nonroot

# Install required certificates for HTTPS requests if needed (reqwest rustls)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary
COPY --from=builder /app/target/release/polarizer /app/polarizer

# Copy the ONNX Runtime dynamic libraries
COPY --from=builder /app/out-libs/ /usr/lib/

# Set the library path so the OS can find libonnxruntime.so
ENV LD_LIBRARY_PATH=/usr/lib

# Copy the downloaded model into the image
COPY --from=builder /app/model_int8.onnx /app/model_int8.onnx

USER nonroot

EXPOSE 9090

ENTRYPOINT ["/app/polarizer"]
