# Redis Stream Contract

Polarizer acts as a consumer of Redis Streams to process image analysis requests, and a producer of Redis Streams to emit results once processing is complete.

## Enqueueing Work (Input Stream)

To dispatch images for analysis, push messages to your configured input Redis stream (e.g., `polarizer:jobs`).

### Payload Schema

```
XADD polarizer:jobs * url "https://cdn.discordapp.com/attachments/.../image.png"
```

The payload should contain a `url` field with the direct HTTP(s) link to the image.

---

## Callbacks (Output Stream)

Once an image is processed (or hits the cache), Polarizer writes a callback event to the configured callback stream (e.g., `polarizer:results`).

### Callback Schema

```
XADD polarizer:results * url "https://..." phash "dGVzdA==" score "0.87" cache_hit "false" elapsed_ms "142" payload "{...}"
```

The payload will contain the following stringified fields:
- `url`: The original URL processed.
- `phash`: The computed perceptual hash (base64 encoded).
- `score`: The model confidence score (0.0 to 1.0).
- `cache_hit`: "true" or "false" depending on whether inference was skipped.
- `elapsed_ms`: Wall-clock processing time in milliseconds.
- `payload`: A JSON string containing all the above fields for convenience.

Example `payload` JSON:
```json
{
  "url": "https://cdn.discordapp.com/attachments/.../image.png",
  "phash": "dGVzdA==",
  "score": 0.87,
  "cache_hit": false,
  "elapsed_ms": 142
}
```
