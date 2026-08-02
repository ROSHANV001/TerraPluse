@echo off
title TerraPulse AI v2.0 - GIS Edition
color 0A

echo.
echo  ============================================
echo   TerraPulse AI v2.0 - GIS Edition
echo   Farm: Aurangabad, Maharashtra, India
echo  ============================================
echo.

if not exist "terrapulse_recommender.pkl" (
    echo  [!] WARNING: Saved models not found
    echo  [!] Run: python save_models.py first
    echo.
    pause
) else (
    echo  [OK] Saved models found - instant startup!
)

if not exist "terrapulse_gis.py" (
    echo  [!] WARNING: terrapulse_gis.py not found
    echo  [!] Place all files in the same folder
    echo.
    pause
) else (
    echo  [OK] GIS module found - weather enabled!
)

echo.
echo  Starting TerraPulse AI GIS Backend...
echo.
echo  Dashboard : http://localhost:8000/dashboard
echo  Weather   : http://localhost:8000/weather
echo  Forecast  : http://localhost:8000/forecast
echo  API Docs  : http://localhost:8000/docs
echo  Status    : http://localhost:8000/status
echo.
echo  Keep this window open while using TerraPulse.
echo  Press Ctrl+C to stop the server.
echo  ============================================
echo.

python terrapulse_backend_gis.py

echo.
echo  [!] Server stopped.
pause
