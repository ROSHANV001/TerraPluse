"""
TerraPulse AI — One-Time Model Saver
======================================
Run this script ONCE after training in Jupyter.
It saves the trained models to disk so the system
starts instantly every time — no retraining needed.

Run:
    python save_models.py

After this you never need to retrain again unless
you want to improve the model with new data.
"""

import os
import pickle
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ModelSaver")

def save_models():

    # ── Step 1: Check dataset exists ─────────────────────────
    if not os.path.exists("terrapulse_dataset_35k.csv"):
        log.error("terrapulse_dataset_35k.csv not found!")
        log.error("Place the dataset in the same folder as this script.")
        return

    # ── Step 2: Train the models ──────────────────────────────
    log.info("Loading dataset...")
    df = pd.read_csv("terrapulse_dataset_35k.csv")
    log.info(f"Dataset loaded: {len(df):,} records")

    log.info("Initialising TerraPulse AI...")
    from terrapulse_ai import TerraPulseAI
    ai = TerraPulseAI()

    log.info("Training models — this takes about 60 seconds...")
    log.info("(This is the LAST time you will ever wait for training)")
    ai.train(df)

    # ── Step 3: Save to disk ──────────────────────────────────
    log.info("Saving Crop Recommender model...")
    with open("terrapulse_recommender.pkl", "wb") as f:
        pickle.dump(ai.recommender, f)

    log.info("Saving Yield Predictor model...")
    with open("terrapulse_predictor.pkl", "wb") as f:
        pickle.dump(ai.predictor, f)

    # ── Step 4: Verify saved files ────────────────────────────
    r_size = os.path.getsize("terrapulse_recommender.pkl") / 1024 / 1024
    p_size = os.path.getsize("terrapulse_predictor.pkl")   / 1024 / 1024

    log.info("=" * 50)
    log.info("MODELS SAVED SUCCESSFULLY")
    log.info(f"  terrapulse_recommender.pkl  {r_size:.1f} MB")
    log.info(f"  terrapulse_predictor.pkl    {p_size:.1f} MB")
    log.info("=" * 50)
    log.info("From now on — double-click START_TERRAPULSE.bat")
    log.info("System will start in under 3 seconds every time.")

if __name__ == "__main__":
    save_models()
