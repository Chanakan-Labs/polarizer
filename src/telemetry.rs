use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// Initialise the global tracing subscriber.
///
/// * In development (`RUST_LOG` unset or `LOG_FORMAT != json`): pretty, human-readable output.
/// * In production (`LOG_FORMAT=json`): structured JSON lines to stdout for log aggregation.
pub fn init_tracing(default_level: &str) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(default_level));

    let is_json = std::env::var("LOG_FORMAT")
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or(false);

    if is_json {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().json().flatten_event(true))
            .init();
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().pretty())
            .init();
    }
}
