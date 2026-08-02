/**
 * TerraPulse AI — Website API Client
 * ====================================
 * Drop this file into your existing website.
 * Call TerraPulse.analyze() to get AI predictions.
 *
 * Your backend must be running:
 *   python terrapulse_backend.py
 *
 * Usage example:
 *   const result = await TerraPulse.analyze({
 *     zone_id: 1, N: 55, P: 40, K: 38,
 *     pH: 5.9, moisture: 42, temperature: 23
 *   });
 *   console.log(result.recommended_crop); // "wheat"
 *   console.log(result.pump);             // "ON"
 */

const TerraPulse = (() => {

  // ── Config ──────────────────────────────────────────────
  // Change this if your backend runs on a different port
  const BASE_URL = 'http://localhost:8000';

  // ── Core API call ────────────────────────────────────────
  async function _post(endpoint, body) {
    const res = await fetch(`${BASE_URL}${endpoint}`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `API error ${res.status}`);
    }
    return res.json();
  }

  async function _get(endpoint) {
    const res = await fetch(`${BASE_URL}${endpoint}`);
    if (!res.ok) throw new Error(`API error ${res.status}`);
    return res.json();
  }

  // ── Public methods ────────────────────────────────────────

  /**
   * analyze(payload) — Main method
   *
   * Send sensor data, get back AI recommendation.
   *
   * @param {Object} payload
   * @param {number} payload.zone_id     - Zone 1, 2, or 3
   * @param {number} payload.N           - Nitrogen mg/kg
   * @param {number} payload.P           - Phosphorus mg/kg
   * @param {number} payload.K           - Potassium mg/kg
   * @param {number} payload.pH          - Soil pH
   * @param {number} payload.moisture    - Moisture %
   * @param {number} payload.temperature - Temperature °C
   *
   * @returns {Promise<Object>} result
   * @returns {string}  result.recommended_crop   - e.g. "wheat"
   * @returns {number}  result.confidence         - e.g. 94.1
   * @returns {Array}   result.top_3_candidates   - [{crop, confidence}]
   * @returns {number}  result.predicted_yield    - e.g. 3.14
   * @returns {number}  result.yield_efficiency   - e.g. 89.8
   * @returns {string}  result.pump               - "ON" or "OFF"
   * @returns {string}  result.mix                - "water"|"fertilizer"|"pH_up"|"pH_down"|"none"
   * @returns {string}  result.reason             - human-readable explanation
   */
  async function analyze(payload) {
    return _post('/analyze', payload);
  }

  /**
   * getAllZones() — Get latest status for all 3 zones
   *
   * Call this to refresh your dashboard.
   * Returns empty object for zones with no data yet.
   */
  async function getAllZones() {
    return _get('/zones');
  }

  /**
   * getZone(zoneId) — Get latest status for one zone
   */
  async function getZone(zoneId) {
    return _get(`/zone/${zoneId}`);
  }

  /**
   * getZoneHistory(zoneId) — Get last 20 readings for a zone
   *
   * Use this to plot trend charts on your website.
   */
  async function getZoneHistory(zoneId) {
    return _get(`/zone/${zoneId}/history`);
  }

  /**
   * getCropProfiles() — All 10 crop profiles
   *
   * Returns ideal NPK, pH, moisture, temp, and base yield per crop.
   */
  async function getCropProfiles() {
    return _get('/crops');
  }

  /**
   * getStatus() — System health check
   *
   * Use this to show "AI online" / "offline" on your site.
   */
  async function getStatus() {
    return _get('/status');
  }

  /**
   * isOnline() — Simple boolean health check
   */
  async function isOnline() {
    try {
      const s = await getStatus();
      return s.model_trained === true;
    } catch {
      return false;
    }
  }

  // ── Expose public API ─────────────────────────────────────
  return { analyze, getAllZones, getZone, getZoneHistory, getCropProfiles, getStatus, isOnline };

})();


// ════════════════════════════════════════════════════════════
//  USAGE EXAMPLES — copy these into your website code
// ════════════════════════════════════════════════════════════

/*
// ── Example 1: Analyze a single zone and show result ──────
async function onSensorDataReceived(sensorData) {
  try {
    const result = await TerraPulse.analyze({
      zone_id:     sensorData.zone,
      N:           sensorData.nitrogen,
      P:           sensorData.phosphorus,
      K:           sensorData.potassium,
      pH:          sensorData.ph,
      moisture:    sensorData.moisture,
      temperature: sensorData.temperature,
    });

    // Show on your page
    document.getElementById('crop-name').textContent    = result.recommended_crop;
    document.getElementById('confidence').textContent   = result.confidence + '%';
    document.getElementById('yield').textContent        = result.predicted_yield + ' t/ha';
    document.getElementById('pump-status').textContent  = result.pump;
    document.getElementById('mix-type').textContent     = result.mix;
    document.getElementById('ai-reason').textContent    = result.reason;

    // Change pump indicator color
    const pumpEl = document.getElementById('pump-indicator');
    pumpEl.style.color = result.pump === 'ON' ? '#4ade80' : '#6b7280';

  } catch (err) {
    console.error('TerraPulse error:', err.message);
  }
}

// ── Example 2: Auto-refresh all zones every 10 seconds ────
async function startDashboard() {
  async function refresh() {
    const online = await TerraPulse.isOnline();
    document.getElementById('ai-status').textContent = online ? '🟢 Online' : '🔴 Offline';

    if (!online) return;

    const { zones } = await TerraPulse.getAllZones();
    for (const [zoneId, data] of Object.entries(zones)) {
      if (!data || !data.recommended_crop) continue;
      document.getElementById(`zone-${zoneId}-crop`).textContent  = data.recommended_crop;
      document.getElementById(`zone-${zoneId}-yield`).textContent = data.predicted_yield + ' t/ha';
      document.getElementById(`zone-${zoneId}-pump`).textContent  = data.pump;
    }
  }

  refresh();                         // run immediately
  setInterval(refresh, 10_000);      // then every 10 seconds
}

startDashboard();

// ── Example 3: Plot yield history chart (Chart.js) ────────
async function plotZoneHistory(zoneId, chartCanvas) {
  const { history } = await TerraPulse.getZoneHistory(zoneId);

  const labels = history.map(h => h.timestamp.slice(11, 19));
  const yields = history.map(h => h.predicted_yield);

  new Chart(chartCanvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: `Zone ${zoneId} Yield (t/ha)`,
        data:  yields,
        borderColor: '#4ade80',
        tension: 0.4,
        fill: false,
      }]
    }
  });
}
*/
