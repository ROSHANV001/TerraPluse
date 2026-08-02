"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TerraPulse AI — FastAPI Backend Server                            ║
║   Connects terrapulse_ai.py → REST API → Your Website Dashboard            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Install:
    pip install fastapi uvicorn paho-mqtt

Run:
    python terrapulse_backend.py

API runs at:  http://localhost:8000
Dashboard at: http://localhost:8000/dashboard
API docs at:  http://localhost:8000/docs  (auto-generated)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator

# ── TerraPulse AI Engine ──────────────────────────────────────────────────────
from terrapulse_ai import (
    SensorReading,
    SyntheticDataGenerator,
    TerraPulseAI,
    IDEAL_CROP_PROFILES,
    CROP_BASE_YIELD,
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TerraPulse-API")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE  (in-memory store for live zone data)
# ─────────────────────────────────────────────────────────────────────────────
zone_latest:  dict[int, dict] = {1: {}, 2: {}, 3: {}}   # latest result per zone
zone_history: dict[int, deque] = {                        # last 20 readings per zone
    1: deque(maxlen=20),
    2: deque(maxlen=20),
    3: deque(maxlen=20),
}
system_stats = {
    "total_requests": 0,
    "started_at": datetime.now().isoformat(),
    "model_trained": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TerraPulse AI API",
    description="Precision Agriculture — Crop Recommendation, Yield Prediction & Actuation",
    version="1.0.0",
)

# Allow your website (any origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS  (request / response schemas)
# ─────────────────────────────────────────────────────────────────────────────

class SensorPayload(BaseModel):
    """What your website or ESP32 sends to the API."""
    zone_id:     int            = Field(..., ge=1, le=3,    description="Zone number 1-3")
    N:           Optional[float]= Field(None, ge=0, le=300, description="Nitrogen mg/kg")
    P:           Optional[float]= Field(None, ge=0, le=200, description="Phosphorus mg/kg")
    K:           Optional[float]= Field(None, ge=0, le=300, description="Potassium mg/kg")
    pH:          Optional[float]= Field(None, ge=3.5, le=9, description="Soil pH")
    moisture:    Optional[float]= Field(None, ge=0, le=100, description="Moisture %")
    temperature: Optional[float]= Field(None, ge=-5, le=55, description="Temperature °C")

    @validator("zone_id")
    def zone_must_be_valid(cls, v):
        if v not in (1, 2, 3):
            raise ValueError("zone_id must be 1, 2, or 3")
        return v


class AnalysisResponse(BaseModel):
    """What the API sends back to your website."""
    zone_id:           int
    timestamp:         str
    recommended_crop:  str
    confidence:        float
    top_3_candidates:  list[dict]
    predicted_yield:   float
    yield_unit:        str
    yield_efficiency:  float
    pump:              str
    mix:               str
    reason:            str
    smoothed_features: dict
    status:            str = "ok"


# ─────────────────────────────────────────────────────────────────────────────
# AI ENGINE  (trained once at startup)
# ─────────────────────────────────────────────────────────────────────────────
ai = TerraPulseAI()

def train_on_startup():
    log.info("Training AI models on startup...")
    try:
        import pickle, os
        # Load saved models if they exist (instant)
        if os.path.exists("terrapulse_recommender.pkl") and \
           os.path.exists("terrapulse_predictor.pkl"):
            with open("terrapulse_recommender.pkl", "rb") as f:
                ai.recommender = pickle.load(f)
            with open("terrapulse_predictor.pkl", "rb") as f:
                ai.predictor = pickle.load(f)
            ai._trained = True
            log.info("Loaded saved models from disk. Ready instantly.")
        else:
            # Generate data and train fresh
            gen = SyntheticDataGenerator(n_samples_per_crop=3500)
            df  = gen.generate()
            ai.train(df)
            log.info("Models trained from scratch.")
    except Exception as e:
        log.error(f"Training failed: {e}")
    system_stats["model_trained"] = ai._trained

# Train in background so server starts immediately
threading.Thread(target=train_on_startup, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>TerraPulse AI API</h2>
    <ul>
      <li><a href="/docs">API Docs (Swagger UI)</a></li>
      <li><a href="/dashboard">Live Dashboard</a></li>
      <li><a href="/status">System Status</a></li>
      <li><a href="/crops">Crop Profiles</a></li>
    </ul>
    """

# ── POST /analyze — Main endpoint called by your website ─────────────────────
@app.post("/analyze", response_model=AnalysisResponse)
def analyze(payload: SensorPayload):
    """
    Send sensor data → Get crop recommendation + yield + actuation command.
    This is the main endpoint your website calls.
    """
    if not ai._trained:
        raise HTTPException(
            status_code=503,
            detail="AI models are still training. Try again in 30 seconds."
        )

    system_stats["total_requests"] += 1

    reading = SensorReading(
        zone_id     = payload.zone_id,
        N           = payload.N,
        P           = payload.P,
        K           = payload.K,
        pH          = payload.pH,
        moisture    = payload.moisture,
        temperature = payload.temperature,
    )

    try:
        result = ai.run(reading)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception(f"AI error: {e}")
        raise HTTPException(status_code=500, detail="AI processing error.")

    rec = result["recommendation"]
    yld = result["yield_forecast"]
    cmd = result["actuation_command"]

    response = {
        "zone_id":           payload.zone_id,
        "timestamp":         datetime.now().isoformat(),
        "recommended_crop":  rec["recommended_crop"],
        "confidence":        round(rec["confidence"] * 100, 1),
        "top_3_candidates":  [
            {"crop": c["crop"], "confidence": round(c["confidence"] * 100, 1)}
            for c in rec["top_3_candidates"]
        ],
        "predicted_yield":   yld["predicted_yield"],
        "yield_unit":        "tonnes/hectare",
        "yield_efficiency":  yld["yield_efficiency"],
        "pump":              cmd["pump"],
        "mix":               cmd["mix"],
        "reason":            cmd["reason"],
        "smoothed_features": result["smoothed_features"],
        "status":            "ok",
    }

    # Store in memory for dashboard
    zone_latest[payload.zone_id]  = response
    zone_history[payload.zone_id].append(response)

    return response


# ── GET /zones — All 3 zones latest status ────────────────────────────────────
@app.get("/zones")
def get_all_zones():
    """Returns latest reading for all 3 zones. Called by live dashboard."""
    return {
        "zones":     zone_latest,
        "timestamp": datetime.now().isoformat(),
    }


# ── GET /zone/{zone_id} — Single zone ────────────────────────────────────────
@app.get("/zone/{zone_id}")
def get_zone(zone_id: int):
    if zone_id not in (1, 2, 3):
        raise HTTPException(status_code=404, detail="Zone must be 1, 2, or 3")
    data = zone_latest.get(zone_id, {})
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No data yet for Zone {zone_id}. Send a POST /analyze first."
        )
    return data


# ── GET /zone/{zone_id}/history — Historical readings ────────────────────────
@app.get("/zone/{zone_id}/history")
def get_zone_history(zone_id: int):
    if zone_id not in (1, 2, 3):
        raise HTTPException(status_code=404, detail="Zone must be 1, 2, or 3")
    return {
        "zone_id": zone_id,
        "count":   len(zone_history[zone_id]),
        "history": list(zone_history[zone_id]),
    }


# ── GET /crops — All ideal crop profiles ─────────────────────────────────────
@app.get("/crops")
def get_crop_profiles():
    """Returns all 10 crop profiles used for training."""
    return {
        crop: {
            **profile,
            "base_yield_t_ha": CROP_BASE_YIELD[crop],
        }
        for crop, profile in IDEAL_CROP_PROFILES.items()
    }


# ── GET /status — System health ───────────────────────────────────────────────
@app.get("/status")
def get_status():
    return {
        "status":         "online",
        "model_trained":  ai._trained,
        "total_requests": system_stats["total_requests"],
        "started_at":     system_stats["started_at"],
        "zones_active":   sum(1 for v in zone_latest.values() if v),
        "timestamp":      datetime.now().isoformat(),
    }


# ── GET /dashboard — Built-in live dashboard HTML page ───────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serves the built-in live farm dashboard."""
    with open("terrapulse_dashboard.html") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting TerraPulse AI Backend...")
    log.info("Dashboard → http://localhost:8000/dashboard")
    log.info("API Docs  → http://localhost:8000/docs")
    uvicorn.run(
        "terrapulse_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",
    )
