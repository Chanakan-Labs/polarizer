# Polarizer

High-performance image analysis microservice built in Rust. Processes images through an ONNX inference pipeline with perceptual hashing and Redis-backed caching.

## Architecture

```
Redis Stream ──▶ Consumer ──▶ Download ──▶ Decode ──▶ pHash ──▶ Cache Check
                                                                    │
                                                          ┌─────────┴─────────┐
                                                          ▼                   ▼
                                                     Cache Hit           Cache Miss
                                                          │                   │
                                                          │            ONNX Inference
                                                          │                   │
                                                          │            Cache Write
                                                          │                   │
                                                          └─────────┬─────────┘
                                                                    ▼
                                                           Results Stream
```

## Quick Start

### Prerequisites

- Rust 1.80+ (stable)
- Redis 7+
- An ONNX model file

### Setup

```bash
# Clone and enter the project
cd polarizer

# Copy environment template
cp .env.example .env
# Edit .env with your Redis URI and model path

# Build and run
cargo run --release
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_URI` | ✅ | — | Redis connection string |
| `MODEL_PATH` | ✅ | — | Path to `.onnx` model file |
| `STREAM_KEY` | | `polarizer:jobs` | Input Redis stream |
| `CONSUMER_GROUP` | | `polarizer-workers` | Consumer group name |
| `CONSUMER_NAME` | | hostname | Unique consumer identifier |
| `WORKER_COUNT` | | `4` | Concurrent processing tasks |
| `BATCH_SIZE` | | `10` | Messages per XREADGROUP call |
| `BLOCK_TIMEOUT_MS` | | `5000` | XREADGROUP block timeout |
| `MAX_DOWNLOAD_BYTES` | | `20971520` | Max download size (20 MiB) |
| `PHASH_PREFIX` | | `phash:` | Redis key prefix for hash cache |
| `PHASH_CACHE_TTL_SECS` | | `604800` | Cache TTL (7 days) |
| `RESULT_STREAM_KEY` | | `polarizer:results` | Output Redis stream |
| `HEALTH_PORT` | | `9090` | Health server port |
| `LOG_LEVEL` | | `info` | Tracing filter directive |
| `LOG_FORMAT` | | `pretty` | Set to `json` for structured output |

### Health Endpoints

| Endpoint | Purpose | Healthy Status |
|----------|---------|----------------|
| `GET /healthz` | Liveness probe | `200 OK` always |
| `GET /readyz` | Readiness probe | `200` when ready, `503` during init |

## Integration Contract

Polarizer communicates purely over Redis Streams using JSON payloads.
For the complete schema detailing how to enqueue tasks and consume callbacks, see [CONTRACT.md](CONTRACT.md).

## Docker

```bash
# Build
docker build -t polarizer .

# Run (mount your model)
docker run -d \
  -v ./model.onnx:/app/model.onnx \
  -e REDIS_URI=redis://host.docker.internal:6379 \
  -e MODEL_PATH=/app/model.onnx \
  -p 9090:9090 \
  polarizer
```

## Project Structure

```
src/
├── main.rs              # Entrypoint: wiring, shutdown signals
├── config.rs            # Typed config from env vars
├── error.rs             # Domain error types (PipelineError)
├── telemetry.rs         # Tracing / logging initialization
├── health.rs            # Axum health server (/healthz, /readyz)
├── redis_stream.rs      # Redis Streams consumer (XREADGROUP)
└── pipeline/
    ├── mod.rs           # Pipeline orchestrator
    ├── download.rs      # Async image downloader with size limits
    ├── hasher.rs        # Perceptual hashing (DCT-based pHash)
    └── inference.rs     # ONNX model loading & tensor preprocessing
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.
