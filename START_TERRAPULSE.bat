@echo off
title TerraPulse AI — Starting...
color 0A

echo.
echo  ============================================
echo   TerraPulse AI — Smart Farm System
echo  ============================================
echo.

:: ── Step 1: Check if saved models exist ─────────────────────
if not exist "terrapulse_recommender.pkl" (
    echo  [!] WARNING: terrapulse_recommender.pkl not found
    echo  [!] AI will retrain from scratch — takes 60 seconds
    echo  [!] Run Jupyter notebook first to save models
    echo.
    pause
) else (
    echo  [OK] Saved models found — instant startup!
)

if not exist "terrapulse_dataset_35k.csv" (
    echo  [!] WARNING: terrapulse_dataset_35k.csv not found
    echo  [!] Place the dataset CSV in this folder
    echo.
    pause
)

echo.
echo  Starting TerraPulse AI Backend...
echo  Dashboard  : http://localhost:8000/dashboard
echo  API Docs   : http://localhost:8000/docs
echo  API Status : http://localhost:8000/status
echo.
echo  Keep this window open while using TerraPulse.
echo  Press Ctrl+C to stop the server.
echo  ============================================
echo.

:: ── Step 2: Start the backend (loads pkl models instantly) ───
python terrapulse_backend.py

:: ── If server crashes, pause so you can read the error ───────
echo.
echo  [!] Server stopped. See error above.
pause
