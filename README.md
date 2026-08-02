# TerraPulse 🌱

**Precision agriculture platform for autonomous smart farm monitoring and control.**

TerraPulse monitors and autonomously controls a 3-zone smart farm using IoT sensors, machine learning, and automated irrigation. Built for a farm in the Aurangabad region of Maharashtra, India (19.881444°N, 75.400667°E).

## What it does

- Reads real-time soil NPK, pH, moisture, and temperature data from each zone via RS485 sensors on ESP32 nodes
- Feeds sensor data through a Kalman filter for noise smoothing
- Uses a Random Forest Classifier to recommend optimal crops per zone
- Uses an XGBoost Regressor to predict expected yield
- Makes autonomous pump/fertilizer actuation decisions using a 15% deviation threshold engine
- Pulls live weather + 5-day rainfall forecasts (OpenWeatherMap GIS) to override irrigation when rain is expected
- Serves everything through a FastAPI backend and a live auto-refreshing dashboard

## Architecture

```
ESP32 Zone Nodes (NPK/pH/moisture/temp sensors)
        │  RS485
        ▼
MQTT Bridge (mqtt_bridge.py)
        │
        ▼
AI Engine (terrapulse_ai.py)
  ├─ Kalman filter (sensor smoothing)
  ├─ Random Forest (crop recommendation)
  ├─ XGBoost (yield prediction)
  └─ Actuation logic (15% deviation threshold, 30s cooldown)
        │
        ▼
FastAPI Backend (terrapulse_backend_gis.py)
  ├─ 7+ REST endpoints
  └─ OpenWeatherMap GIS integration
        │
        ▼
Live Dashboard (terrapulse_dashboard.html)
```

## Key design decisions

- **15% deviation threshold** drives pump actuation — balances responsiveness with stability, avoiding overreaction to sensor noise
- **30-second cooldown** between zone commands protects pump motors from rapid on/off cycling
- **Priority waterfall** for actuation decisions: moisture → pH → NPK
- **GIS weather override**: irrigation is suppressed when rain is detected in the forecast, calibrated for Aurangabad's seasonal rainfall
- Models are trained once and saved as `.pkl` files, so the backend starts instantly in production without retraining

## Repo structure

| File | Purpose |
|---|---|
| `terrapulse_ai.py` | AI engine — crop recommendation, yield prediction, actuation logic |
| `esp32_zone_node.ino` | ESP32 firmware — sensor reading + MQTT publishing |
| `mqtt_bridge.py` | Bridges MQTT sensor data into the AI engine |
| `terrapulse_backend_gis.py` | FastAPI backend with GIS/weather integration |
| `terrapulse_dashboard.html` | Live monitoring dashboard |
| `*.ipynb` | Training and evaluation notebook (29 cells) |
| `*.bat` | One-click Windows startup scripts |
| `terrapulse_integration_package.zip` | Handover package with API reference for frontend/backend teams |

## Setup

1. Clone the repo
2. Create a `.env` file with your OpenWeatherMap API key (see `.env.example`)
3. Install Python dependencies: `pip install -r requirements.txt`
4. Run the backend: `python terrapulse_backend_gis.py` (or use the provided `.bat` script on Windows)
5. Flash `esp32_zone_node.ino` to each ESP32 zone node via Arduino IDE
6. Open `terrapulse_dashboard.html` to view live zone data

## Hardware

- ESP32 dev boards (one per zone)
- NPK RS485 sensors
- Capacitive soil moisture sensors
- DS18B20 temperature sensors
- pH sensors with ADS1115 ADC
- Relay modules for pump/fertilizer control

## Status

Core system built and confirmed working end-to-end with saved models. Next steps: physical wiring of hardware to zones for full autonomous operation, and frontend/backend team integration.

## Team ownership

- **Backend team**: server, model files, API endpoints
- **Frontend team**: JS client integration with dashboard/API
- **Hardware team**: ESP32 firmware flashing, sensor wiring

---
Built by Roshan Nitin Vyavhare.
