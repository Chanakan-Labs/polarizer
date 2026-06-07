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

The `score` is the softmax probability for the *predicted label* (which is output in the `label` field, e.g. `nsfw` or `normal`). A higher score means higher confidence that the image matches the predicted label.

For example, if `label` is `nsfw` and `score` is `0.95`, the model is 95% confident the image is NSFW. If `label` is `normal` and `score` is `0.85`, the model is 85% confident the image is normal (which means the probability of it being NSFW is `0.15`).

| Confidence Score | Interpretation |
|------------------|----------------|
| `0.8 – 1.0` | High confidence in the predicted label |
| `0.5 – 0.8` | Lower confidence — may warrant manual review if borderline |

> **Note:** These thresholds are guidelines. Tune the decision boundary based on your moderation policy.
