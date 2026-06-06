# ── Builder stage ────────────────────────────────────────────────────────────
FROM rust:1.87-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dependency compilation by copying manifests first.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src

# Build the actual binary.
COPY src/ src/
RUN touch src/main.rs && cargo build --release

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r polarizer && useradd -r -g polarizer polarizer

WORKDIR /app
COPY --from=builder /app/target/release/polarizer /app/polarizer

# Model is expected to be mounted or baked into a derived image.
# COPY model.onnx /app/model.onnx

USER polarizer

EXPOSE 9090

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD curl -sf http://localhost:9090/healthz || exit 1

ENTRYPOINT ["/app/polarizer"]
