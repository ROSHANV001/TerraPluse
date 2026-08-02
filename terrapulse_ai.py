"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              TerraPulse AI — Precision Agriculture ML Engine                ║
║     Crop Recommendation · Yield Prediction · Autonomous Actuation Logic     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │  IoT Sensors (ESP32) → Noisy Raw Data                │
  │       ↓                                              │
  │  DataSmoother   (Moving Average / Kalman)            │
  │       ↓                                              │
  │  Validator      (bounds check, null guard)           │
  │       ↓                                              │
  │  CropRecommender  (RandomForest Classifier)          │
  │  YieldPredictor   (XGBoost Regressor)                │
  │       ↓                                              │
  │  SoilHealthMonitor → JSON Relay Command              │
  │      {"pump": "ON", "zone": 1, "mix": "fertilizer"} │
  └──────────────────────────────────────────────────────┘

Author  : TerraPulse AI System
Version : 2.0.0
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TerraPulse")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  IDEAL CROP PROFILES
#     Ground truth reference for the actuation engine.
#     All units: N/P/K in mg/kg, pH dimensionless, moisture in %, temp in °C.
# ─────────────────────────────────────────────────────────────────────────────

IDEAL_CROP_PROFILES: dict[str, dict[str, float]] = {
    "rice":      {"N": 80,  "P": 40,  "K": 40,  "pH": 6.0, "moisture": 70, "temperature": 25},
    "wheat":     {"N": 60,  "P": 60,  "K": 40,  "pH": 6.5, "moisture": 50, "temperature": 20},
    "maize":     {"N": 80,  "P": 40,  "K": 20,  "pH": 6.5, "moisture": 55, "temperature": 22},
    "cotton":    {"N": 120, "P": 40,  "K": 40,  "pH": 6.5, "moisture": 45, "temperature": 28},
    "sugarcane": {"N": 100, "P": 50,  "K": 50,  "pH": 6.5, "moisture": 65, "temperature": 28},
    "soybean":   {"N": 20,  "P": 60,  "K": 40,  "pH": 6.5, "moisture": 50, "temperature": 24},
    "chickpea":  {"N": 40,  "P": 60,  "K": 80,  "pH": 6.5, "moisture": 35, "temperature": 18},
    "tomato":    {"N": 100, "P": 80,  "K": 100, "pH": 6.2, "moisture": 60, "temperature": 24},
    "potato":    {"N": 120, "P": 60,  "K": 120, "pH": 5.5, "moisture": 65, "temperature": 18},
    "banana":    {"N": 100, "P": 75,  "K": 50,  "pH": 6.0, "moisture": 75, "temperature": 27},
}

# Baseline yield in tonnes/hectare (used as regression target anchor)
CROP_BASE_YIELD: dict[str, float] = {
    "rice": 4.5, "wheat": 3.5, "maize": 5.5, "cotton": 2.2,
    "sugarcane": 70.0, "soybean": 2.8, "chickpea": 1.5,
    "tomato": 35.0, "potato": 25.0, "banana": 40.0,
}

# Physical validity bounds — anything outside is a sensor fault
SENSOR_BOUNDS: dict[str, tuple[float, float]] = {
    "N":           (0,   300),
    "P":           (0,   200),
    "K":           (0,   300),
    "pH":          (3.5, 9.0),
    "moisture":    (0,   100),
    "temperature": (-5,  55),
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SensorReading:
    """Raw sensor payload from an ESP32 node."""
    zone_id:     int
    N:           Optional[float] = None
    P:           Optional[float] = None
    K:           Optional[float] = None
    pH:          Optional[float] = None
    moisture:    Optional[float] = None
    temperature: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "zone_id"}


@dataclass
class ActuationCommand:
    """Hardware relay command serialisable to JSON."""
    zone:   int
    pump:   str   = "OFF"         # "ON" | "OFF"
    mix:    str   = "none"        # "water" | "fertilizer" | "pH_up" | "pH_down" | "npk_blend"
    reason: str   = ""
    params: dict  = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA SMOOTHER  (Moving Average + Kalman Filter)
# ─────────────────────────────────────────────────────────────────────────────

class DataSmoother:
    """
    Two-stage noise reduction for IoT sensor streams.

    Stage 1 — Moving Average: removes high-frequency spikes.
    Stage 2 — 1D Kalman Filter: tracks the true underlying signal
               using a process model and measurement noise estimate.

    Each sensor channel maintains its own independent state.
    """

    def __init__(self, window: int = 5, kalman_q: float = 1e-5, kalman_r: float = 0.01):
        self.window    = window
        self.q         = kalman_q   # process noise covariance
        self.r         = kalman_r   # measurement noise covariance
        self._buffers:  dict[str, deque]  = {}
        self._kalman:   dict[str, dict]   = {}

    def _init_channel(self, key: str, value: float) -> None:
        self._buffers[key] = deque(maxlen=self.window)
        self._kalman[key]  = {"x_hat": value, "P": 1.0}

    def _moving_average(self, key: str, value: float) -> float:
        self._buffers[key].append(value)
        return float(np.mean(self._buffers[key]))

    def _kalman_update(self, key: str, z: float) -> float:
        state = self._kalman[key]
        # Predict
        x_prior = state["x_hat"]
        P_prior  = state["P"] + self.q
        # Update
        K        = P_prior / (P_prior + self.r)
        state["x_hat"] = x_prior + K * (z - x_prior)
        state["P"]     = (1 - K) * P_prior
        return state["x_hat"]

    def smooth(self, zone_id: int, feature: str, raw_value: float) -> float:
        key = f"z{zone_id}_{feature}"
        if key not in self._buffers:
            self._init_channel(key, raw_value)
        ma_val = self._moving_average(key, raw_value)
        return round(self._kalman_update(key, ma_val), 4)

    def smooth_reading(self, reading: SensorReading) -> SensorReading:
        """Apply smoothing to all non-null fields in a SensorReading."""
        smoothed = SensorReading(zone_id=reading.zone_id)
        for feat, val in reading.to_dict().items():
            if val is not None:
                object.__setattr__(smoothed, feat, self.smooth(reading.zone_id, feat, val))
            else:
                object.__setattr__(smoothed, feat, None)
        return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class SensorValidator:
    """
    Validates a SensorReading against physical bounds and null constraints.
    Returns a cleaned dict or raises descriptive errors — never crashes silently.
    """

    def validate(self, reading: SensorReading) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        errors:  list[str]        = []

        for feature, (lo, hi) in SENSOR_BOUNDS.items():
            val = getattr(reading, feature, None)

            if val is None:
                errors.append(f"[Zone {reading.zone_id}] '{feature}' is missing (None)")
                continue

            if not isinstance(val, (int, float)) or np.isnan(val):
                errors.append(f"[Zone {reading.zone_id}] '{feature}' is non-numeric: {val!r}")
                continue

            if not (lo <= val <= hi):
                errors.append(
                    f"[Zone {reading.zone_id}] '{feature}' out of bounds: "
                    f"{val} ∉ [{lo}, {hi}]"
                )
                continue

            cleaned[feature] = float(val)

        if errors:
            log.warning("Validation issues:\n  " + "\n  ".join(errors))

        if not cleaned:
            raise ValueError(
                f"Zone {reading.zone_id}: All sensor values are invalid. "
                "Cannot proceed without any clean features."
            )

        return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SYNTHETIC DATASET GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Generates a labelled dataset by sampling from Gaussian distributions
    centred on each crop's ideal profile. Adds realistic sensor noise.
    Yields a regression target from a physics-inspired formula:
        yield = base_yield × (1 − Σ deviation_penalties)
    """

    def __init__(self, n_samples_per_crop: int = 200, noise_level: float = 0.12,
                 random_state: int = 42):
        self.n        = n_samples_per_crop
        self.noise    = noise_level
        self.rng      = np.random.default_rng(random_state)

    def _yield_formula(self, crop: str, sample: dict[str, float]) -> float:
        ideal = IDEAL_CROP_PROFILES[crop]
        base  = CROP_BASE_YIELD[crop]
        penalty = 0.0
        for feat, ideal_val in ideal.items():
            if ideal_val == 0:
                continue
            deviation = abs(sample[feat] - ideal_val) / ideal_val
            penalty  += min(deviation, 1.0) * 0.12   # max 12% per feature
        return round(max(base * (1 - penalty), base * 0.20), 3)

    def generate(self) -> pd.DataFrame:
        records = []
        for crop, profile in IDEAL_CROP_PROFILES.items():
            for _ in range(self.n):
                sample: dict[str, Any] = {"crop": crop}
                for feat, ideal_val in profile.items():
                    sigma = ideal_val * self.noise if ideal_val != 0 else 0.5
                    raw   = self.rng.normal(ideal_val, sigma)
                    lo, hi = SENSOR_BOUNDS[feat]
                    sample[feat] = float(np.clip(raw, lo, hi))

                sample["yield_t_ha"] = self._yield_formula(crop, sample)
                sample["zone_id"]    = self.rng.integers(1, 4)
                records.append(sample)

        df = pd.DataFrame(records)
        log.info(f"Generated {len(df)} synthetic records for {len(IDEAL_CROP_PROFILES)} crops.")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CROP RECOMMENDER  (Random Forest Classifier)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = ["N", "P", "K", "pH", "moisture", "temperature"]

class CropRecommender:
    """
    Multi-class classifier that maps soil/climate features to the optimal crop.
    Uses calibrated probabilities so the caller can surface a confidence score.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.model   = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler  = StandardScaler()
        self.encoder = LabelEncoder()
        self._trained = False

    def fit(self, df: pd.DataFrame) -> None:
        X = self.scaler.fit_transform(df[FEATURE_COLS])
        y = self.encoder.fit_transform(df["crop"])
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_tr, y_tr)
        self._trained = True

        y_pred = self.model.predict(X_val)
        log.info("\n" + classification_report(
            y_val, y_pred, target_names=self.encoder.classes_
        ))

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        if not self._trained:
            raise RuntimeError("CropRecommender must be trained before inference.")

        row = np.array([[features.get(f, 0) for f in FEATURE_COLS]])
        row_scaled = self.scaler.transform(row)

        class_idx  = self.model.predict(row_scaled)[0]
        proba      = self.model.predict_proba(row_scaled)[0]

        top3_idx   = np.argsort(proba)[::-1][:3]
        top3       = [
            {"crop": self.encoder.classes_[i], "confidence": round(float(proba[i]), 4)}
            for i in top3_idx
        ]

        return {
            "recommended_crop": self.encoder.classes_[class_idx],
            "confidence":       round(float(proba[class_idx]), 4),
            "top_3_candidates": top3,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  YIELD PREDICTOR  (XGBoost Regressor)
# ─────────────────────────────────────────────────────────────────────────────

class YieldPredictor:
    """
    Regresses expected yield (t/ha) from soil features + crop label.
    XGBoost is used for its robustness to feature interactions.
    """

    def __init__(self, random_state: int = 42):
        self.model  = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=0,
        )
        self.scaler  = StandardScaler()
        self.encoder = LabelEncoder()
        self._trained = False
        self._feat_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        df = df.copy()
        df["crop_enc"] = self.encoder.fit_transform(df["crop"])
        self._feat_cols = FEATURE_COLS + ["crop_enc"]

        X = self.scaler.fit_transform(df[self._feat_cols])
        y = df["yield_t_ha"].values

        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        self._trained = True

        y_pred = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2  = r2_score(y_val, y_pred)
        log.info(f"YieldPredictor — MAE: {mae:.3f} t/ha | R²: {r2:.4f}")

    def predict(self, features: dict[str, float], crop: str) -> dict[str, Any]:
        if not self._trained:
            raise RuntimeError("YieldPredictor must be trained before inference.")

        try:
            crop_enc = self.encoder.transform([crop])[0]
        except ValueError:
            raise ValueError(f"Unknown crop '{crop}'. Fit the model first.")

        row = np.array([[features.get(f, 0) for f in FEATURE_COLS] + [crop_enc]])
        row_scaled = self.scaler.transform(row)
        predicted  = float(self.model.predict(row_scaled)[0])

        return {
            "crop":              crop,
            "predicted_yield":   round(predicted, 3),
            "unit":              "tonnes/hectare",
            "base_yield":        CROP_BASE_YIELD.get(crop, "N/A"),
            "yield_efficiency":  round(predicted / CROP_BASE_YIELD[crop] * 100, 1)
                                 if crop in CROP_BASE_YIELD else "N/A",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SOIL HEALTH MONITOR & ACTUATION ENGINE
#
#  Decision Logic (the "15% Rule"):
#
#  For each parameter in [moisture, N, P, K, pH]:
#    delta = (ideal_val - current_val) / ideal_val
#
#    if delta > 0.15  →  parameter is DEFICIENT  → trigger correction
#    if delta < -0.15 →  parameter is EXCESSIVE   → log warning, no pump
#
#  Priority waterfall:
#    1. moisture deficient  → pump="ON", mix="water"
#    2. pH deficient        → pump="ON", mix="pH_up"
#    3. pH excessive        → pump="ON", mix="pH_down"
#    4. N or P or K deficient → pump="ON", mix="fertilizer" (or npk_blend if >1)
#    5. all within range    → pump="OFF", mix="none"
# ─────────────────────────────────────────────────────────────────────────────

DEVIATION_THRESHOLD = 0.15   # 15% below ideal triggers actuation

class SoilHealthMonitor:
    """
    Prescriptive decision engine.
    Compares real-time sensor values against the ideal profile for the
    recommended crop, flags deficiencies, and emits JSON relay commands.
    """

    def check_soil_health(
        self,
        zone_id:  int,
        features: dict[str, float],
        crop:     str,
    ) -> ActuationCommand:
        """
        Core method — returns an ActuationCommand with all fields set.

        Parameters
        ----------
        zone_id  : Farm zone (1–3)
        features : Validated, smoothed sensor dict
        crop     : Recommended crop (drives ideal profile selection)
        """
        if crop not in IDEAL_CROP_PROFILES:
            raise ValueError(f"No ideal profile found for crop '{crop}'.")

        ideal     = IDEAL_CROP_PROFILES[crop]
        deficient = []
        excessive = []
        deviations: dict[str, float] = {}

        for param, ideal_val in ideal.items():
            current = features.get(param)
            if current is None or ideal_val == 0:
                continue

            delta = (ideal_val - current) / ideal_val   # positive → below ideal
            deviations[param] = round(delta, 4)

            if delta > DEVIATION_THRESHOLD:
                deficient.append(param)
                log.warning(
                    f"Zone {zone_id} | '{param}' DEFICIENT: "
                    f"current={current:.2f}, ideal={ideal_val:.2f}, "
                    f"shortfall={delta*100:.1f}%"
                )
            elif delta < -DEVIATION_THRESHOLD:
                excessive.append(param)
                log.info(
                    f"Zone {zone_id} | '{param}' EXCESSIVE: "
                    f"current={current:.2f}, ideal={ideal_val:.2f}"
                )

        # ── Priority Waterfall ────────────────────────────────────────────────
        cmd = ActuationCommand(zone=zone_id, params={"deviations": deviations})

        if "moisture" in deficient:
            cmd.pump   = "ON"
            cmd.mix    = "water"
            cmd.reason = (
                f"Soil moisture critically low "
                f"({features.get('moisture', '?'):.1f}% vs ideal {ideal['moisture']}%). "
                "Activating irrigation pump."
            )

        elif "pH" in deficient:
            cmd.pump   = "ON"
            cmd.mix    = "pH_up"
            cmd.reason = (
                f"Soil pH too low ({features.get('pH', '?'):.2f} vs ideal {ideal['pH']}). "
                "Injecting alkaline solution."
            )

        elif "pH" in excessive:
            cmd.pump   = "ON"
            cmd.mix    = "pH_down"
            cmd.reason = (
                f"Soil pH too high ({features.get('pH', '?'):.2f} vs ideal {ideal['pH']}). "
                "Injecting acidic solution."
            )

        else:
            npk_deficient = [p for p in ("N", "P", "K") if p in deficient]
            if len(npk_deficient) > 1:
                cmd.pump   = "ON"
                cmd.mix    = "npk_blend"
                cmd.reason = (
                    f"Multiple macro-nutrients deficient: {npk_deficient}. "
                    "Deploying balanced NPK fertilizer blend."
                )
            elif len(npk_deficient) == 1:
                nutrient   = npk_deficient[0]
                cmd.pump   = "ON"
                cmd.mix    = "fertilizer"
                cmd.reason = (
                    f"Nutrient '{nutrient}' deficient "
                    f"({features.get(nutrient, '?'):.1f} vs ideal {ideal[nutrient]}). "
                    f"Injecting {nutrient}-rich fertilizer."
                )
            else:
                cmd.pump   = "OFF"
                cmd.mix    = "none"
                cmd.reason = (
                    f"All parameters within ±{DEVIATION_THRESHOLD*100:.0f}% of "
                    f"ideal profile for '{crop}'. No actuation required."
                )

        cmd.params["deficient"] = deficient
        cmd.params["excessive"] = excessive
        return cmd


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ORCHESTRATOR  — ties all components together
# ─────────────────────────────────────────────────────────────────────────────

class TerraPulseAI:
    """
    Top-level orchestrator.

    Usage:
        ai = TerraPulseAI()
        ai.train(dataframe)
        result = ai.run(SensorReading(zone_id=1, N=55, P=35, K=38,
                                      pH=5.9, moisture=48, temperature=24))
    """

    def __init__(self):
        self.smoother   = DataSmoother(window=5)
        self.validator  = SensorValidator()
        self.recommender = CropRecommender()
        self.predictor  = YieldPredictor()
        self.monitor    = SoilHealthMonitor()
        self._trained   = False
        log.info("TerraPulse AI initialised.")

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> None:
        log.info("═" * 60)
        log.info("Training CropRecommender …")
        self.recommender.fit(df)
        log.info("Training YieldPredictor …")
        self.predictor.fit(df)
        self._trained = True
        log.info("All models trained. System ready.")
        log.info("═" * 60)

    # ── Inference Pipeline ───────────────────────────────────────────────────

    def run(self, raw_reading: SensorReading) -> dict[str, Any]:
        """
        Full inference pipeline for a single sensor reading.

        Returns
        -------
        dict with keys:
            zone_id, smoothed_features, recommendation,
            yield_forecast, actuation_command
        """
        if not self._trained:
            raise RuntimeError("Call train() before run().")

        # Stage 1: Smooth
        smoothed = self.smoother.smooth_reading(raw_reading)
        log.info(f"Zone {raw_reading.zone_id} | Smoothed reading applied.")

        # Stage 2: Validate
        features = self.validator.validate(smoothed)

        # Stage 3: Recommend crop
        rec = self.recommender.predict(features)
        best_crop = rec["recommended_crop"]
        log.info(f"Zone {raw_reading.zone_id} | Recommended crop: {best_crop} "
                 f"(confidence: {rec['confidence']*100:.1f}%)")

        # Stage 4: Predict yield
        yld = self.predictor.predict(features, best_crop)
        log.info(f"Zone {raw_reading.zone_id} | Predicted yield: "
                 f"{yld['predicted_yield']} t/ha ({yld['yield_efficiency']}% efficiency)")

        # Stage 5: Actuation decision
        cmd = self.monitor.check_soil_health(raw_reading.zone_id, features, best_crop)
        log.info(f"Zone {raw_reading.zone_id} | Actuation: pump={cmd.pump}, mix={cmd.mix}")

        return {
            "zone_id":           raw_reading.zone_id,
            "smoothed_features": features,
            "recommendation":    rec,
            "yield_forecast":    yld,
            "actuation_command": json.loads(cmd.to_json()),
        }

    def run_all_zones(
        self, readings: list[SensorReading]
    ) -> list[dict[str, Any]]:
        """Convenience wrapper to process all zones in one call."""
        results = []
        for reading in readings:
            try:
                results.append(self.run(reading))
            except Exception as exc:
                log.error(f"Zone {reading.zone_id} failed: {exc}")
                results.append({"zone_id": reading.zone_id, "error": str(exc)})
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 10.  DEMO / ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _pretty_print(result: dict) -> None:
    zone = result.get("zone_id", "?")
    if "error" in result:
        print(f"\n{'─'*60}\n⚠  Zone {zone} ERROR: {result['error']}")
        return

    rec = result["recommendation"]
    yld = result["yield_forecast"]
    cmd = result["actuation_command"]

    print(f"""
{'═'*60}
 Zone {zone}  |  {rec['recommended_crop'].upper()}  ({rec['confidence']*100:.1f}% confidence)
{'─'*60}
 Top 3 Candidates : {
     " → ".join(f"{c['crop']} ({c['confidence']*100:.0f}%)"
                for c in rec['top_3_candidates'])}
 Predicted Yield  : {yld['predicted_yield']} t/ha  ({yld['yield_efficiency']}% efficiency)
 Base Yield Ref   : {yld['base_yield']} t/ha
{'─'*60}
 ACTUATION DECISION
   Pump            : {cmd['pump']}
   Fertilizer Mix  : {cmd['mix']}
   Reason          : {cmd['reason']}
{'─'*60}
 Raw JSON Command →
{json.dumps({k: cmd[k] for k in ('zone','pump','mix','reason')}, indent=4)}
{'═'*60}""")


def main() -> None:
    # ── Step 1: Generate training data
    gen = SyntheticDataGenerator(n_samples_per_crop=300)
    df  = gen.generate()

    # ── Step 2: Initialise and train
    ai = TerraPulseAI()
    ai.train(df)

    # ── Step 3: Simulate real-time zone readings ──────────────────────────────
    #   Zone 1: Moisture-starved wheat field
    #   Zone 2: Nitrogen-deficient maize field
    #   Zone 3: pH imbalance (too acidic) — rice
    #   Zone 4: Sensor fault injection (out-of-bounds pH + missing K)
    # ─────────────────────────────────────────────────────────────────────────

    test_readings = [
        SensorReading(zone_id=1, N=58, P=57, K=39, pH=6.4, moisture=38.0, temperature=19),
        SensorReading(zone_id=2, N=52, P=38, K=18, pH=6.6, moisture=54.0, temperature=22),
        SensorReading(zone_id=3, N=76, P=38, K=37, pH=4.8, moisture=67.0, temperature=25),
        SensorReading(zone_id=1, N=80, P=60, K=None, pH=15.5, moisture=50.0, temperature=20),
    ]

    results = ai.run_all_zones(test_readings)
    for r in results:
        _pretty_print(r)


if __name__ == "__main__":
    main()
