use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::{Json, Router, extract::State, routing::get};
use serde::Serialize;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

/// Shared, lock-free health state.
pub struct HealthState {
    ready: AtomicBool,
}

impl HealthState {
    pub fn new() -> Self {
        Self {
            ready: AtomicBool::new(false),
        }
    }

    pub fn set_ready(&self, val: bool) {
        self.ready.store(val, Ordering::Release);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    ready: bool,
}

async fn liveness() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        ready: true,
    })
}

async fn readiness(
    State(state): State<Arc<HealthState>>,
) -> (axum::http::StatusCode, Json<HealthResponse>) {
    let ready = state.is_ready();
    let status_code = if ready {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    };

    (
        status_code,
        Json(HealthResponse {
            status: if ready { "ready" } else { "not_ready" },
            version: env!("CARGO_PKG_VERSION"),
            ready,
        }),
    )
}

/// Run a lightweight HTTP server with `/healthz` and `/readyz` endpoints.
pub async fn serve(port: u16, state: Arc<HealthState>, cancel: CancellationToken) {
    let app = Router::new()
        .route("/healthz", get(liveness))
        .route("/readyz", get(readiness))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            error!(error = %e, addr, "failed to bind health server");
            return;
        }
    };

    info!(addr, "health server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(cancel.cancelled_owned())
        .await
        .unwrap_or_else(|e| error!(error = %e, "health server error"));
}
