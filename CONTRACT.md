# Redis Stream Contract

Polarizer acts as a consumer of Redis Streams to process image analysis requests, and a producer of Redis Streams to emit results once processing is complete.

## Enqueueing Work (Input Stream)

To dispatch images for analysis, push messages to your configured input Redis stream (e.g., `polarizer:jobs`).

### Payload Schema

```
XADD polarizer:jobs * url "https://cdn.discordapp.com/attachments/.../image.png"
```

The stream entry must contain a `url` field with the direct HTTP(s) link to the image.

---

## Callbacks (Output Stream)

Once an image is processed (or hits the cache), Polarizer writes a result to the configured output stream (e.g., `polarizer:results`).

### Callback Schema

```
XADD polarizer:results *
  url        "https://cdn.discordapp.com/attachments/.../image.png"
  phash      "dGVzdA=="
  score      "0.87"
  label      "nsfw"
  cache_hit  "false"
  elapsed_ms "142"
  payload    "{...}"
```

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | The original URL that was processed |
| `phash` | string | Perceptual hash of the image (base64-encoded) |
| `score` | string (float) | Probability for the target label after softmax (0.0–1.0) |
| `label` | string | Human-readable label name (e.g. `nsfw`, `sfw`) |
| `cache_hit` | string (bool) | `true` if the result came from pHash cache, `false` if inference ran |
| `elapsed_ms` | string (u64) | Wall-clock processing time in milliseconds |
| `payload` | string (JSON) | All fields above serialized as a JSON object for convenience |

### Example `payload` JSON

```json
{
  "url": "https://cdn.discordapp.com/attachments/.../image.png",
  "phash": "dGVzdA==",
  "score": 0.87,
  "label": "nsfw",
  "cache_hit": false,
  "elapsed_ms": 142
}
```

### Interpreting the Score

The `score` is the softmax probability for the target label (configured via `MODEL_TARGET_LABEL_INDEX`, default `1` = `nsfw`). A higher score means higher confidence that the image matches that label.

| Score Range | Interpretation |
|-------------|----------------|
| `0.0 – 0.3` | Very likely SFW |
| `0.3 – 0.7` | Ambiguous — may warrant manual review |
| `0.7 – 1.0` | Very likely NSFW |

> **Note:** These thresholds are guidelines. Tune the decision boundary based on your moderation policy.
