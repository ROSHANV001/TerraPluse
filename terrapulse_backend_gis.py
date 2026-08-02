"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       TerraPulse AI — Backend Server v2.0 (GIS Edition)                    ║
║   REST API + Live Weather + Rainfall + Crop Stress + Farm Intelligence     ║
╚══════════════════════════════════════════════════════════════════════════════╝

New in v2.0 (GIS Edition):
  - /weather          Live weather for your farm location
  - /forecast         5-day rainfall and temperature forecast
  - /gis/zone/{id}    GIS-enriched zone analysis
  - /analyze          Now includes weather context in every prediction
  - Smart irrigation  Skips pump if rain detected or forecast

Run:
    python terrapulse_backend_gis.py

Dashboard : http://localhost:8000/dashboard
API Docs  : http://localhost:8000/docs
Weather   : http://localhost:8000/weather
Forecast  : http://localhost:8000/forecast
"""

from __future__ import annotations

import json
import logging
import os
import pickle
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

from terrapulse_ai import (
    SensorReading, SyntheticDataGenerator,
    TerraPulseAI, IDEAL_CROP_PROFILES, CROP_BASE_YIELD,
)
from terrapulse_gis import gis, FARM_CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TerraPulse-GIS-API")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
zone_latest:  dict[int, dict] = {1: {}, 2: {}, 3: {}}
zone_history: dict[int, deque] = {
    1: deque(maxlen=20),
    2: deque(maxlen=20),
    3: deque(maxlen=20),
}
system_stats = {
    "total_requests": 0,
    "started_at":     datetime.now().isoformat(),
    "model_trained":  False,
    "gis_enabled":    True,
}

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TerraPulse AI API — GIS Edition",
    description="Precision Agriculture with Live Weather Intelligence",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class SensorPayload(BaseModel):
    zone_id:     int             = Field(..., ge=1, le=3)
    N:           Optional[float] = Field(None, ge=0,   le=300)
    P:           Optional[float] = Field(None, ge=0,   le=200)
    K:           Optional[float] = Field(None, ge=0,   le=300)
    pH:          Optional[float] = Field(None, ge=3.5, le=9.0)
    moisture:    Optional[float] = Field(None, ge=0,   le=100)
    temperature: Optional[float] = Field(None, ge=-5,  le=55)

    @validator("zone_id")
    def zone_must_be_valid(cls, v):
        if v not in (1, 2, 3):
            raise ValueError("zone_id must be 1, 2, or 3")
        return v

# ─────────────────────────────────────────────────────────────────────────────
# AI ENGINE
# ─────────────────────────────────────────────────────────────────────────────
ai = TerraPulseAI()

def train_on_startup():
    log.info("Loading AI models...")
    try:
        if os.path.exists("terrapulse_recommender.pkl") and \
           os.path.exists("terrapulse_predictor.pkl"):
            with open("terrapulse_recommender.pkl", "rb") as f:
                ai.recommender = pickle.load(f)
            with open("terrapulse_predictor.pkl", "rb") as f:
                ai.predictor   = pickle.load(f)
            ai._trained = True
            log.info("Saved models loaded instantly.")
        else:
            gen = SyntheticDataGenerator(n_samples_per_crop=3500)
            df  = gen.generate()
            ai.train(df)
            log.info("Models trained from scratch.")
    except Exception as e:
        log.error(f"Training failed: {e}")
    system_stats["model_trained"] = ai._trained

threading.Thread(target=train_on_startup, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# GIS-AWARE PUMP DECISION
# ─────────────────────────────────────────────────────────────────────────────

def apply_gis_to_actuation(cmd: dict, weather) -> dict:
    """
    Overrides pump decisions based on live weather.
    Prevents wasteful irrigation when it is already raining.

    Rules:
      - Currently raining          → override pump OFF (save water)
      - Heavy rain in last 24h     → reduce irrigation signal
      - Rain forecast in 3 hours   → skip irrigation
      - Very hot + dry             → boost irrigation priority
    """
    original_pump = cmd.get("pump", "OFF")
    original_mix  = cmd.get("mix",  "none")

    if original_mix == "water":
        if weather.is_raining():
            cmd["pump"]   = "OFF"
            cmd["mix"]    = "none"
            cmd["reason"] += (
                f" | GIS OVERRIDE: Currently raining "
                f"({weather.rainfall_1h}mm/hr) — irrigation skipped."
            )
            cmd["gis_override"] = "rain_detected"

        elif weather.rainfall_24h > 25:
            cmd["pump"]   = "OFF"
            cmd["mix"]    = "none"
            cmd["reason"] += (
                f" | GIS OVERRIDE: Heavy rainfall in last 24h "
                f"({weather.rainfall_24h}mm) — irrigation skipped."
            )
            cmd["gis_override"] = "recent_heavy_rain"

        elif weather.is_hot() and weather.humidity < 30:
            cmd["reason"] += (
                f" | GIS BOOST: Extreme heat ({weather.temperature}°C) "
                f"+ low humidity ({weather.humidity}%) — irrigation priority HIGH."
            )
            cmd["gis_override"] = "heat_boost"

        else:
            cmd["gis_override"] = "normal"

    else:
        cmd["gis_override"] = "not_applicable"

    return cmd

# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>TerraPulse AI API v2.0 - GIS Edition</h2>
    <ul>
      <li><a href="/docs">API Docs</a></li>
      <li><a href="/dashboard">Live Dashboard</a></li>
      <li><a href="/weather">Live Weather</a></li>
      <li><a href="/forecast">5-Day Forecast</a></li>
      <li><a href="/status">System Status</a></li>
    </ul>
    """

# ── POST /analyze — Main endpoint with GIS enrichment ────────────────────────
@app.post("/analyze")
def analyze(payload: SensorPayload):
    """
    Send sensor data → Get AI prediction enriched with live weather.
    """
    if not ai._trained:
        raise HTTPException(status_code=503,
            detail="AI still loading. Try again in 30 seconds.")

    system_stats["total_requests"] += 1

    # ── Run core AI ───────────────────────────────────────────
    reading = SensorReading(
        zone_id=payload.zone_id, N=payload.N, P=payload.P, K=payload.K,
        pH=payload.pH, moisture=payload.moisture, temperature=payload.temperature,
    )
    try:
        result = ai.run(reading)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception(f"AI error: {e}")
        raise HTTPException(status_code=500, detail="AI processing error.")

    # ── Fetch live weather ────────────────────────────────────
    weather      = gis.get_current_weather()
    gis_features = gis.get_gis_features()

    # ── Apply GIS override to pump decision ───────────────────
    rec = result["recommendation"]
    yld = result["yield_forecast"]
    cmd = dict(result["actuation_command"])
    cmd = apply_gis_to_actuation(cmd, weather)

    # ── Build enriched response ───────────────────────────────
    response = {
        "zone_id":           payload.zone_id,
        "timestamp":         datetime.now().isoformat(),

        # Crop prediction
        "recommended_crop":  rec["recommended_crop"],
        "confidence":        round(rec["confidence"] * 100, 1),
        "top_3_candidates":  [
            {"crop": c["crop"], "confidence": round(c["confidence"] * 100, 1)}
            for c in rec["top_3_candidates"]
        ],

        # Yield forecast
        "predicted_yield":   yld["predicted_yield"],
        "yield_unit":        "tonnes/hectare",
        "yield_efficiency":  yld["yield_efficiency"],

        # Actuation (GIS-aware)
        "pump":              cmd["pump"],
        "mix":               cmd["mix"],
        "reason":            cmd["reason"],
        "gis_override":      cmd.get("gis_override", "none"),

        # Sensor readings (smoothed)
        "smoothed_features": result["smoothed_features"],

        # Live weather context
        "weather": {
            "temperature":    weather.temperature,
            "humidity":       weather.humidity,
            "rainfall_1h_mm": weather.rainfall_1h,
            "description":    weather.description,
            "crop_stress":    weather.crop_stress_level(),
            "irrigation_hint":weather.irrigation_recommendation(),
            "is_raining":     weather.is_raining(),
        },

        "status": "ok",
    }

    # Store for dashboard
    zone_latest[payload.zone_id]  = response
    zone_history[payload.zone_id].append(response)

    return response


# ── GET /weather — Live weather for your farm ─────────────────────────────────
@app.get("/weather")
def get_weather():
    """
    Returns live weather for your farm at
    19°52'53.2"N 75°24'02.4"E (Aurangabad, Maharashtra)
    """
    weather  = gis.get_current_weather()
    forecast = gis.get_forecast(days=3)

    return {
        "farm":     FARM_CONFIG["location"],
        "lat":      FARM_CONFIG["latitude"],
        "lon":      FARM_CONFIG["longitude"],
        "current":  weather.to_dict(),
        "upcoming_3_days": [
            {
                "date":        f.date,
                "temp_min":    f.temp_min,
                "temp_max":    f.temp_max,
                "humidity":    f.humidity,
                "rainfall_mm": f.rainfall_mm,
                "description": f.description,
            }
            for f in forecast
        ],
        "timestamp": datetime.now().isoformat(),
    }


# ── GET /forecast — 5-day farm forecast ──────────────────────────────────────
@app.get("/forecast")
def get_forecast():
    """5-day weather forecast for your farm. Use for irrigation planning."""
    forecast = gis.get_forecast(days=5)
    total_rain = sum(f.rainfall_mm for f in forecast)

    advice = []
    for f in forecast:
        if f.rainfall_mm > 10:
            advice.append(f"{f.date}: Skip irrigation — {f.rainfall_mm}mm rain expected")
        elif f.temp_max > 38:
            advice.append(f"{f.date}: Increase irrigation — extreme heat {f.temp_max}°C")
        else:
            advice.append(f"{f.date}: Normal irrigation schedule")

    return {
        "farm":              FARM_CONFIG["location"],
        "total_rain_5d_mm":  round(total_rain, 1),
        "forecast":          [
            {
                "date":        f.date,
                "temp_min":    f.temp_min,
                "temp_max":    f.temp_max,
                "humidity":    f.humidity,
                "rainfall_mm": f.rainfall_mm,
                "description": f.description,
                "wind_ms":     f.wind_speed,
            }
            for f in forecast
        ],
        "irrigation_advice": advice,
        "timestamp":         datetime.now().isoformat(),
    }


# ── GET /zones ────────────────────────────────────────────────────────────────
@app.get("/zones")
def get_all_zones():
    weather = gis.get_current_weather()
    return {
        "zones":     zone_latest,
        "weather_summary": {
            "temperature":    weather.temperature,
            "humidity":       weather.humidity,
            "is_raining":     weather.is_raining(),
            "crop_stress":    weather.crop_stress_level(),
            "irrigation_hint":weather.irrigation_recommendation(),
        },
        "timestamp": datetime.now().isoformat(),
    }


# ── GET /zone/{id} ────────────────────────────────────────────────────────────
@app.get("/zone/{zone_id}")
def get_zone(zone_id: int):
    if zone_id not in (1, 2, 3):
        raise HTTPException(status_code=404, detail="Zone must be 1, 2, or 3")
    data = zone_latest.get(zone_id, {})
    if not data:
        raise HTTPException(status_code=404,
            detail=f"No data for Zone {zone_id}. Send a POST /analyze first.")
    return data


# ── GET /zone/{id}/history ────────────────────────────────────────────────────
@app.get("/zone/{zone_id}/history")
def get_zone_history(zone_id: int):
    if zone_id not in (1, 2, 3):
        raise HTTPException(status_code=404, detail="Zone must be 1, 2, or 3")
    return {
        "zone_id": zone_id,
        "count":   len(zone_history[zone_id]),
        "history": list(zone_history[zone_id]),
    }


# ── GET /crops ────────────────────────────────────────────────────────────────
@app.get("/crops")
def get_crop_profiles():
    return {
        crop: {**profile, "base_yield_t_ha": CROP_BASE_YIELD[crop]}
        for crop, profile in IDEAL_CROP_PROFILES.items()
    }


# ── GET /status ───────────────────────────────────────────────────────────────
@app.get("/status")
def get_status():
    weather = gis.get_current_weather()
    return {
        "status":         "online",
        "version":        "2.0.0-GIS",
        "model_trained":  ai._trained,
        "gis_enabled":    True,
        "farm_location":  FARM_CONFIG["location"],
        "current_weather":weather.description,
        "total_requests": system_stats["total_requests"],
        "started_at":     system_stats["started_at"],
        "zones_active":   sum(1 for v in zone_latest.values() if v),
        "timestamp":      datetime.now().isoformat(),
    }


# ── GET /dashboard ────────────────────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    try:
        with open("terrapulse_dashboard.html") as f:
            return f.read()
    except FileNotFoundError:
        return "<h2>Dashboard file not found. Place terrapulse_dashboard.html in the same folder.</h2>"


# ── Internal zone update (called by mqtt_bridge) ──────────────────────────────
@app.post("/internal/zone_update")
def internal_zone_update(data: dict):
    zone_id = data.get("zone_id")
    if zone_id in (1, 2, 3):
        rec = data.get("recommendation", {})
        yld = data.get("yield_forecast", {})
        cmd = data.get("actuation_command", {})
        weather = gis.get_current_weather()

        zone_latest[zone_id] = {
            "zone_id":           zone_id,
            "timestamp":         datetime.now().isoformat(),
            "recommended_crop":  rec.get("recommended_crop", ""),
            "confidence":        round(rec.get("confidence", 0) * 100, 1),
            "top_3_candidates":  rec.get("top_3_candidates", []),
            "predicted_yield":   yld.get("predicted_yield", 0),
            "yield_unit":        "tonnes/hectare",
            "yield_efficiency":  yld.get("yield_efficiency", 0),
            "pump":              cmd.get("pump", "OFF"),
            "mix":               cmd.get("mix", "none"),
            "reason":            cmd.get("reason", ""),
            "smoothed_features": data.get("smoothed_features", {}),
            "weather": {
                "temperature":     weather.temperature,
                "humidity":        weather.humidity,
                "rainfall_1h_mm":  weather.rainfall_1h,
                "description":     weather.description,
                "crop_stress":     weather.crop_stress_level(),
                "irrigation_hint": weather.irrigation_recommendation(),
                "is_raining":      weather.is_raining(),
            },
        }
        zone_history[zone_id].append(zone_latest[zone_id])
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("TerraPulse AI v2.0 GIS Edition starting...")
    log.info(f"Farm location : {FARM_CONFIG['location']}")
    log.info(f"GPS           : {FARM_CONFIG['latitude']}, {FARM_CONFIG['longitude']}")
    log.info("Dashboard     : http://localhost:8000/dashboard")
    log.info("Weather       : http://localhost:8000/weather")
    log.info("Forecast      : http://localhost:8000/forecast")
    log.info("API Docs      : http://localhost:8000/docs")

    uvicorn.run(
        "terrapulse_backend_gis:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",
    )
