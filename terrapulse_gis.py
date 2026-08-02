"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         TerraPulse AI — GIS Weather Integration Module                      ║
║   Live weather + rainfall + humidity fetched for your exact farm location   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Farm Location : 19°52'53.2"N  75°24'02.4"E  (Aurangabad region, Maharashtra)
Weather API   : OpenWeatherMap
Data refresh  : Every 10 minutes (cached to avoid API rate limits)

Author  : TerraPulse AI System
Version : 2.0.0  (GIS Edition)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file, if present

log = logging.getLogger("TerraPulse-GIS")

# ─────────────────────────────────────────────────────────────────────────────
# FARM CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_api_key = os.getenv("OPENWEATHER_API_KEY")
if not _api_key:
    raise RuntimeError(
        "OPENWEATHER_API_KEY environment variable is not set. "
        "Create a .env file (see .env.example) or set it in your shell/OS "
        "environment before starting TerraPulse."
    )

FARM_CONFIG = {
    "name":        "TerraPulse Farm",
    "location":    "Aurangabad, Maharashtra, India",
    "latitude":    19.881444,
    "longitude":   75.400667,
    "api_key":     _api_key,
    "timezone":    "Asia/Kolkata",
}

# Cache weather data for 10 minutes — avoids hitting API every sensor read
CACHE_DURATION_S = 600

# ─────────────────────────────────────────────────────────────────────────────
# GIS DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WeatherData:
    """Live weather snapshot for the farm location."""
    temperature:    float = 0.0   # °C
    humidity:       float = 0.0   # %
    rainfall_1h:    float = 0.0   # mm in last 1 hour
    rainfall_24h:   float = 0.0   # mm in last 24 hours
    wind_speed:     float = 0.0   # m/s
    wind_direction: float = 0.0   # degrees
    cloud_cover:    float = 0.0   # %
    pressure:       float = 0.0   # hPa
    uv_index:       float = 0.0   # UV index
    description:    str   = ""    # e.g. "light rain"
    feels_like:     float = 0.0   # °C
    dew_point:      float = 0.0   # °C
    visibility:     float = 0.0   # km
    timestamp:      float = field(default_factory=time.time)

    def is_raining(self) -> bool:
        return self.rainfall_1h > 0.1

    def is_hot(self) -> bool:
        return self.temperature > 35.0

    def is_humid(self) -> bool:
        return self.humidity > 80.0

    def crop_stress_level(self) -> str:
        """Returns low / medium / high stress based on weather."""
        stress = 0
        if self.temperature > 38:  stress += 2
        elif self.temperature > 35: stress += 1
        if self.humidity > 85:     stress += 1
        if self.wind_speed > 10:   stress += 1
        if self.uv_index > 8:      stress += 1
        if stress >= 3: return "high"
        if stress >= 2: return "medium"
        return "low"

    def irrigation_recommendation(self) -> str:
        """
        Weather-aware irrigation hint.
        Used alongside soil moisture sensor readings.
        """
        if self.is_raining():
            return "skip"        # raining — no irrigation needed
        if self.rainfall_24h > 20:
            return "reduce"      # rained recently — reduce irrigation
        if self.temperature > 38 and self.humidity < 30:
            return "increase"    # very hot and dry — increase irrigation
        return "normal"

    def to_dict(self) -> dict:
        return {
            "temperature":          self.temperature,
            "humidity":             self.humidity,
            "rainfall_1h_mm":       self.rainfall_1h,
            "rainfall_24h_mm":      self.rainfall_24h,
            "wind_speed_ms":        self.wind_speed,
            "cloud_cover_pct":      self.cloud_cover,
            "pressure_hpa":         self.pressure,
            "uv_index":             self.uv_index,
            "description":          self.description,
            "feels_like":           self.feels_like,
            "is_raining":           self.is_raining(),
            "crop_stress":          self.crop_stress_level(),
            "irrigation_hint":      self.irrigation_recommendation(),
        }


@dataclass
class ForecastDay:
    """Single day forecast."""
    date:         str
    temp_min:     float
    temp_max:     float
    humidity:     float
    rainfall_mm:  float
    description:  str
    wind_speed:   float


# ─────────────────────────────────────────────────────────────────────────────
# GIS WEATHER FETCHER
# ─────────────────────────────────────────────────────────────────────────────

class GISWeatherFetcher:
    """
    Fetches live weather and 5-day forecast for the farm location.
    Caches results for 10 minutes to avoid rate limiting.
    Falls back to safe defaults if API is unreachable.
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self):
        self._cache:          Optional[WeatherData] = None
        self._cache_time:     float = 0
        self._forecast_cache: list  = []
        self._forecast_time:  float = 0
        self._api_failures:   int   = 0

    # ── Current Weather ───────────────────────────────────────────────────────

    def get_current_weather(self, force_refresh: bool = False) -> WeatherData:
        """
        Returns current weather for the farm.
        Uses cache if data is less than 10 minutes old.
        """
        now = time.time()
        cache_valid = (now - self._cache_time) < CACHE_DURATION_S

        if self._cache and cache_valid and not force_refresh:
            log.debug("Weather data from cache.")
            return self._cache

        try:
            data = self._fetch_current()
            self._cache      = data
            self._cache_time = now
            self._api_failures = 0
            log.info(
                f"Weather updated — {data.temperature}°C, "
                f"humidity {data.humidity}%, "
                f"rain {data.rainfall_1h}mm, "
                f"stress: {data.crop_stress_level()}"
            )
            return data

        except Exception as e:
            self._api_failures += 1
            log.warning(f"Weather fetch failed (attempt {self._api_failures}): {e}")

            # Return cached data if available even if stale
            if self._cache:
                log.info("Using stale cached weather data.")
                return self._cache

            # Return safe defaults so AI keeps working offline
            log.warning("No cache available. Using default weather values.")
            return self._safe_defaults()

    def _fetch_current(self) -> WeatherData:
        """Calls OpenWeatherMap current weather API."""
        url    = f"{self.BASE_URL}/weather"
        params = {
            "lat":   FARM_CONFIG["latitude"],
            "lon":   FARM_CONFIG["longitude"],
            "appid": FARM_CONFIG["api_key"],
            "units": "metric",
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        d = resp.json()

        rain = d.get("rain", {})
        return WeatherData(
            temperature    = round(d["main"]["temp"],       2),
            humidity       = round(d["main"]["humidity"],   2),
            rainfall_1h    = round(rain.get("1h", 0.0),     2),
            rainfall_24h   = round(rain.get("3h", 0.0) * 8, 2),
            wind_speed     = round(d["wind"]["speed"],      2),
            wind_direction = round(d["wind"].get("deg", 0), 1),
            cloud_cover    = round(d["clouds"]["all"],      1),
            pressure       = round(d["main"]["pressure"],   1),
            uv_index       = 0.0,    # requires separate UV API call
            description    = d["weather"][0]["description"],
            feels_like     = round(d["main"]["feels_like"], 2),
            dew_point      = round(
                d["main"]["temp"] - ((100 - d["main"]["humidity"]) / 5), 2
            ),
            visibility     = round(d.get("visibility", 10000) / 1000, 1),
        )

    # ── 5-Day Forecast ────────────────────────────────────────────────────────

    def get_forecast(self, days: int = 5) -> list[ForecastDay]:
        """Returns 5-day weather forecast for the farm location."""
        now = time.time()
        cache_valid = (now - self._forecast_time) < CACHE_DURATION_S * 3

        if self._forecast_cache and cache_valid:
            return self._forecast_cache[:days]

        try:
            url    = f"{self.BASE_URL}/forecast"
            params = {
                "lat":   FARM_CONFIG["latitude"],
                "lon":   FARM_CONFIG["longitude"],
                "appid": FARM_CONFIG["api_key"],
                "units": "metric",
                "cnt":   40,    # 5 days × 8 readings per day
            }
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()

            # Group by day
            days_dict: dict[str, list] = {}
            for item in data["list"]:
                date = item["dt_txt"][:10]
                days_dict.setdefault(date, []).append(item)

            forecast = []
            for date, readings in list(days_dict.items())[:days]:
                temps    = [r["main"]["temp"] for r in readings]
                humidity = [r["main"]["humidity"] for r in readings]
                rain     = sum(r.get("rain", {}).get("3h", 0) for r in readings)
                winds    = [r["wind"]["speed"] for r in readings]
                desc     = readings[len(readings)//2]["weather"][0]["description"]

                forecast.append(ForecastDay(
                    date        = date,
                    temp_min    = round(min(temps), 1),
                    temp_max    = round(max(temps), 1),
                    humidity    = round(sum(humidity)/len(humidity), 1),
                    rainfall_mm = round(rain, 2),
                    description = desc,
                    wind_speed  = round(sum(winds)/len(winds), 1),
                ))

            self._forecast_cache = forecast
            self._forecast_time  = now
            log.info(f"5-day forecast updated for {FARM_CONFIG['location']}")
            return forecast

        except Exception as e:
            log.warning(f"Forecast fetch failed: {e}")
            return []

    # ── GIS Feature Vector ────────────────────────────────────────────────────

    def get_gis_features(self) -> dict:
        """
        Returns weather features formatted for AI model input.
        This is what gets merged with your sensor readings.
        """
        w = self.get_current_weather()
        f = self.get_forecast(days=3)

        # Upcoming rainfall in next 3 days
        upcoming_rain = sum(day.rainfall_mm for day in f) if f else 0.0

        return {
            "weather_temp":       w.temperature,
            "weather_humidity":   w.humidity,
            "rainfall_1h":        w.rainfall_1h,
            "rainfall_24h":       w.rainfall_24h,
            "upcoming_rain_3d":   round(upcoming_rain, 2),
            "wind_speed":         w.wind_speed,
            "cloud_cover":        w.cloud_cover,
            "crop_stress":        w.crop_stress_level(),
            "irrigation_hint":    w.irrigation_recommendation(),
            "is_raining":         w.is_raining(),
        }

    # ── Safe Defaults ─────────────────────────────────────────────────────────

    def _safe_defaults(self) -> WeatherData:
        """
        Returns safe average values for Aurangabad, Maharashtra.
        Used when API is unreachable — AI keeps working offline.
        """
        return WeatherData(
            temperature  = 28.0,
            humidity     = 55.0,
            rainfall_1h  = 0.0,
            rainfall_24h = 0.0,
            wind_speed   = 2.5,
            cloud_cover  = 30.0,
            pressure     = 1013.0,
            uv_index     = 6.0,
            description  = "data unavailable — using defaults",
            feels_like   = 30.0,
            dew_point    = 18.0,
            visibility   = 10.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

# Import this anywhere in the project — one shared instance
gis = GISWeatherFetcher()
