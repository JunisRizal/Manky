#!/usr/bin/env python3
"""
Web-Dashboard für das Kryptowährungs-Vorhersagesystem
Bietet intuitive Benutzeroberfläche und Kontrollfunktionen
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')

from flask import Flask, render_template, jsonify, request, send_from_directory
import json
import time
import threading
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Optional
import psutil

# Import des Hauptsystems
from minimal_data_predictor import MinimalDataPredictionSystem

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'crypto-predictor-dashboard-2025'

# Globale Variablen
prediction_system = None
system_status = {
    'running': False,
    'last_update': None,
    'predictions': {},
    'errors': [],
    'performance_metrics': {}
}

# Konfiguration
DASHBOARD_CONFIG = {
    'auto_refresh_interval': 60,  # Sekunden
    'max_error_history': 50,
    'supported_symbols': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD']
}

class DashboardManager:
    """Verwaltet Dashboard-Funktionalitäten"""
    
    def __init__(self):
        self.prediction_system = None
        self.auto_refresh_thread = None
        self.is_running = False
        
    def initialize_system(self, symbols=None):
        """Initialisiert das Vorhersagesystem"""
        try:
            if symbols is None:
                symbols = DASHBOARD_CONFIG['supported_symbols'][:4]  # Erste 4 für bessere Performance
            
            self.prediction_system = MinimalDataPredictionSystem(symbols)
            self.is_running = True
            
            logger.info(f"System initialisiert mit {len(symbols)} Symbolen")
            return True, f"System erfolgreich initialisiert mit {len(symbols)} Symbolen"
            
        except Exception as e:
            logger.error(f"Fehler bei System-Initialisierung: {e}")
            return False, str(e)
    
    def start_auto_refresh(self):
        """Startet automatische Aktualisierung"""
        if self.auto_refresh_thread and self.auto_refresh_thread.is_alive():
            return
        
        self.auto_refresh_thread = threading.Thread(target=self._auto_refresh_worker, daemon=True)
        self.auto_refresh_thread.start()
        logger.info("Auto-Refresh gestartet")
    
    def _auto_refresh_worker(self):
        """Worker-Thread für automatische Aktualisierung"""
        while self.is_running:
            try:
                if self.prediction_system:
                    self.update_predictions()
                time.sleep(DASHBOARD_CONFIG['auto_refresh_interval'])
            except Exception as e:
                logger.error(f"Fehler im Auto-Refresh: {e}")
                time.sleep(30)  # Kürzere Pause bei Fehlern
    
    def update_predictions(self):
        """Aktualisiert Vorhersagen"""
        try:
            if not self.prediction_system:
                return False, "System nicht initialisiert"
            
            start_time = time.time()
            predictions = self.prediction_system.make_predictions()
            execution_time = time.time() - start_time
            
            # Status aktualisieren
            system_status['predictions'] = {}
            for symbol, prediction in predictions.items():
                system_status['predictions'][symbol] = {
                    'symbol': prediction.symbol,
                    'current_price': prediction.current_price,
                    'predicted_change': prediction.predicted_change * 100,
                    'target_price': prediction.target_price,
                    'confidence': prediction.confidence,
                    'uncertainty': prediction.uncertainty,
                    'data_quality': prediction.data_quality,
                    'features_used': prediction.features_used,
                    'timestamp': prediction.timestamp
                }
            
            system_status['last_update'] = datetime.now().isoformat()
            system_status['performance_metrics']['last_execution_time'] = execution_time
            system_status['running'] = True
            
            logger.info(f"Vorhersagen aktualisiert für {len(predictions)} Symbole in {execution_time:.2f}s")
            return True, f"Vorhersagen für {len(predictions)} Symbole aktualisiert"
            
        except Exception as e:
            error_msg = f"Fehler bei Vorhersage-Update: {e}"
            logger.error(error_msg)
            self._add_error(error_msg)
            return False, error_msg
    
    def get_system_health(self):
        """Gibt System-Gesundheitsstatus zurück"""
        try:
            health_data = {
                'timestamp': time.time(),
                'system_running': self.is_running,
                'predictor_initialized': self.prediction_system is not None,
                'last_update': system_status.get('last_update'),
                'active_predictions': len(system_status.get('predictions', {})),
                'recent_errors': len([e for e in system_status.get('errors', []) 
                                    if time.time() - e.get('timestamp', 0) < 3600]),  # Letzte Stunde
                'system_metrics': self._get_system_metrics()
            }
            
            # Gesamtstatus bestimmen
            if not health_data['system_running']:
                health_data['status'] = 'stopped'
            elif health_data['recent_errors'] > 5:
                health_data['status'] = 'degraded'
            elif health_data['active_predictions'] == 0:
                health_data['status'] = 'warning'
            else:
                health_data['status'] = 'healthy'
            
            return health_data
            
        except Exception as e:
            logger.error(f"Fehler bei Health-Check: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _get_system_metrics(self):
        """Sammelt System-Metriken"""
        try:
            # CPU und Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Disk-Usage
            disk = psutil.disk_usage('/')
            
            # Netzwerk-Test (einfacher Ping)
            import subprocess
            try:
                subprocess.check_output(['ping', '-c', '1', '8.8.8.8'], 
                                      stderr=subprocess.DEVNULL, timeout=5)
                network_ok = True
            except:
                network_ok = False
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / 1024 / 1024,
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                'network_connectivity': network_ok
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Metrik-Sammlung: {e}")
            return {}
    
    def _add_error(self, error_message):
        """Fügt Fehler zur Historie hinzu"""
        error_entry = {
            'timestamp': time.time(),
            'message': error_message,
            'datetime': datetime.now().isoformat()
        }
        
        if 'errors' not in system_status:
            system_status['errors'] = []
        
        system_status['errors'].append(error_entry)
        
        # Begrenzte Historie
        if len(system_status['errors']) > DASHBOARD_CONFIG['max_error_history']:
            system_status['errors'] = system_status['errors'][-DASHBOARD_CONFIG['max_error_history']:]
    
    def stop_system(self):
        """Stoppt das System"""
        self.is_running = False
        system_status['running'] = False
        logger.info("System gestoppt")
        return True, "System gestoppt"

# Globaler Dashboard-Manager
dashboard_manager = DashboardManager()

@app.route('/')
def index():
    """Hauptseite des Dashboards"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """API-Endpoint für System-Status"""
    try:
        health = dashboard_manager.get_system_health()
        
        response_data = {
            'status': 'success',
            'data': {
                'system_health': health,
                'predictions': system_status.get('predictions', {}),
                'last_update': system_status.get('last_update'),
                'performance_metrics': system_status.get('performance_metrics', {}),
                'supported_symbols': DASHBOARD_CONFIG['supported_symbols']
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Fehler in /api/status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predictions')
def api_predictions():
    """API-Endpoint für Vorhersagen"""
    try:
        symbols = request.args.get('symbols', '').split(',') if request.args.get('symbols') else None
        
        predictions = system_status.get('predictions', {})
        
        if symbols:
            symbols = [s.strip().upper() for s in symbols if s.strip()]
            predictions = {k: v for k, v in predictions.items() if k in symbols}
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Fehler in /api/predictions: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/control/start', methods=['POST'])
def api_start_system():
    """Startet das System"""
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols', DASHBOARD_CONFIG['supported_symbols'][:4])
        
        success, message = dashboard_manager.initialize_system(symbols)
        
        if success:
            dashboard_manager.start_auto_refresh()
            return jsonify({'status': 'success', 'message': message})
        else:
            return jsonify({'status': 'error', 'message': message}), 500
            
    except Exception as e:
        logger.error(f"Fehler beim System-Start: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/control/stop', methods=['POST'])
def api_stop_system():
    """Stoppt das System"""
    try:
        success, message = dashboard_manager.stop_system()
        return jsonify({'status': 'success', 'message': message})
        
    except Exception as e:
        logger.error(f"Fehler beim System-Stop: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/control/update', methods=['POST'])
def api_update_predictions():
    """Aktualisiert Vorhersagen manuell"""
    try:
        success, message = dashboard_manager.update_predictions()
        
        if success:
            return jsonify({'status': 'success', 'message': message})
        else:
            return jsonify({'status': 'error', 'message': message}), 500
            
    except Exception as e:
        logger.error(f"Fehler bei manueller Aktualisierung: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/health')
def api_health():
    """Detaillierter Health-Check"""
    try:
        health = dashboard_manager.get_system_health()
        return jsonify(health)
        
    except Exception as e:
        logger.error(f"Fehler im Health-Check: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/logs')
def api_logs():
    """Gibt aktuelle Logs zurück"""
    try:
        max_lines = int(request.args.get('lines', 50))
        
        # Aktuelle Errors aus dem System
        recent_errors = system_status.get('errors', [])[-max_lines:]
        
        # Zusätzlich System-Logs (falls verfügbar)
        log_entries = []
        
        for error in recent_errors:
            log_entries.append({
                'timestamp': error['datetime'],
                'level': 'ERROR',
                'message': error['message']
            })
        
        # Erfolgreiche Updates hinzufügen
        if system_status.get('last_update'):
            log_entries.append({
                'timestamp': system_status['last_update'],
                'level': 'INFO',
                'message': f"Vorhersagen aktualisiert für {len(system_status.get('predictions', {}))} Symbole"
            })
        
        # Nach Timestamp sortieren
        log_entries.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'logs': log_entries[:max_lines],
            'count': len(log_entries)
        })
        
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Logs: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/export')
def api_export():
    """Exportiert aktuelle Daten"""
    try:
        export_format = request.args.get('format', 'json').lower()
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'system_status': {
                'running': system_status.get('running', False),
                'last_update': system_status.get('last_update'),
                'active_predictions': len(system_status.get('predictions', {}))
            },
            'predictions': system_status.get('predictions', {}),
            'performance_metrics': system_status.get('performance_metrics', {}),
            'health': dashboard_manager.get_system_health()
        }
        
        if export_format == 'json':
            return jsonify(export_data)
        else:
            return jsonify({'status': 'error', 'message': 'Nur JSON-Export unterstützt'}), 400
            
    except Exception as e:
        logger.error(f"Fehler beim Export: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# HTML-Template für das Dashboard
@app.route('/templates/dashboard.html')
def dashboard_template():
    """Liefert das Dashboard-HTML-Template"""
    html_content = '''
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Predictor Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .status-indicator {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .status-healthy { background: #2ecc71; color: white; }
        .status-warning { background: #f39c12; color: white; }
        .status-error { background: #e74c3c; color: white; }
        .status-stopped { background: #95a5a6; color: white; }
        
        .controls {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            margin: 5px;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }
        
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #229954; }
        
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        
        .btn-warning { background: #f39c12; }
        .btn-warning:hover { background: #e67e22; }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }
        
        .prediction-item {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #3498db;
        }
        
        .prediction-symbol {
            font-weight: bold;
            font-size: 1.1em;
            color: #2c3e50;
        }
        
        .prediction-price {
            font-size: 1.2em;
            color: #27ae60;
            margin: 5px 0;
        }
        
        .prediction-change {
            font-weight: bold;
            margin: 5px 0;
        }
        
        .change-positive { color: #27ae60; }
        .change-negative { color: #e74c3c; }
        
        .confidence-bar {
            background: #ecf0f1;
            border-radius: 10px;
            height: 8px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
            transition: width 0.3s ease;
        }
        
        .metric-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .metric-item:last-child {
            border-bottom: none;
        }
        
        .metric-value {
            font-weight: bold;
            color: #2c3e50;
        }
        
        .log-entry {
            background: #f8f9fa;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        .log-error { border-left: 4px solid #e74c3c; }
        .log-info { border-left: 4px solid #3498db; }
        .log-warning { border-left: 4px solid #f39c12; }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
        }
        
        .auto-refresh {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
        }
        
        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 34px;
        }
        
        .slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        
        input:checked + .slider {
            background-color: #2196F3;
        }
        
        input:checked + .slider:before {
            transform: translateX(26px);
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .grid {
                grid-template-columns: 1fr;
            }
            
            .controls {
                text-align: center;
            }
            
            .btn {
                display: block;
                width: 100%;
                margin: 5px 0;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Crypto Predictor Dashboard</h1>
            <p>Wissenschaftliches Kryptowährungs-Vorhersagesystem</p>
            <div id="systemStatus" class="status-indicator status-stopped">System gestoppt</div>
            <div style="margin-top: 10px;">
                <small>Letzte Aktualisierung: <span id="lastUpdate">Nie</span></small>
            </div>
        </div>
        
        <div class="controls">
            <h3>🎛️ System-Kontrolle</h3>
            <button class="btn btn-success" onclick="startSystem()">▶️ System starten</button>
            <button class="btn btn-danger" onclick="stopSystem()">⏹️ System stoppen</button>
            <button class="btn btn-warning" onclick="updatePredictions()">🔄 Aktualisieren</button>
            <button class="btn" onclick="exportData()">💾 Daten exportieren</button>
            
            <div class="auto-refresh" style="margin-top: 15px;">
                <label>Auto-Refresh:</label>
                <label class="switch">
                    <input type="checkbox" id="autoRefreshToggle" onchange="toggleAutoRefresh()">
                    <span class="slider"></span>
                </label>
                <span id="autoRefreshStatus">Aus</span>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📈 Aktuelle Vorhersagen</h3>
                <div id="predictionsContainer">
                    <div class="loading">Keine Vorhersagen verfügbar</div>
                </div>
            </div>
            
            <div class="card">
                <h3>💻 System-Metriken</h3>
                <div id="metricsContainer">
                    <div class="loading">Lade Metriken...</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>📋 System-Logs</h3>
            <div id="logsContainer">
                <div class="loading">Lade Logs...</div>
            </div>
        </div>
    </div>

    <script>
        let autoRefreshInterval = null;
        let isAutoRefreshEnabled = false;
        
        // Initialisierung
        document.addEventListener('DOMContentLoaded', function() {
            loadStatus();
            loadLogs();
        });
        
        // Auto-Refresh Toggle
        function toggleAutoRefresh() {
            const toggle = document.getElementById('autoRefreshToggle');
            const status = document.getElementById('autoRefreshStatus');
            
            isAutoRefreshEnabled = toggle.checked;
            
            if (isAutoRefreshEnabled) {
                status.textContent = 'An (60s)';
                autoRefreshInterval = setInterval(loadStatus, 60000);
                loadStatus(); // Sofort laden
            } else {
                status.textContent = 'Aus';
                if (autoRefreshInterval) {
                    clearInterval(autoRefreshInterval);
                    autoRefreshInterval = null;
                }
            }
        }
        
        // System-Kontrolle
        async function startSystem() {
            try {
                showLoading('System wird gestartet...');
                const response = await fetch('/api/control/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    showMessage('✅ ' + result.message, 'success');
                    setTimeout(loadStatus, 2000);
                } else {
                    showMessage('❌ ' + result.message, 'error');
                }
            } catch (error) {
                showMessage('❌ Fehler beim Starten: ' + error.message, 'error');
            }
        }
        
        async function stopSystem() {
            try {
                const response = await fetch('/api/control/stop', { method: 'POST' });
                const result = await response.json();
                
                if (result.status === 'success') {
                    showMessage('✅ ' + result.message, 'success');
                    loadStatus();
                } else {
                    showMessage('❌ ' + result.message, 'error');
                }
            } catch (error) {
                showMessage('❌ Fehler beim Stoppen: ' + error.message, 'error');
            }
        }
        
        async function updatePredictions() {
            try {
                showLoading('Vorhersagen werden aktualisiert...');
                const response = await fetch('/api/control/update', { method: 'POST' });
                const result = await response.json();
                
                if (result.status === 'success') {
                    showMessage('✅ ' + result.message, 'success');
                    loadStatus();
                } else {
                    showMessage('❌ ' + result.message, 'error');
                }
            } catch (error) {
                showMessage('❌ Fehler bei Aktualisierung: ' + error.message, 'error');
            }
        }
        
        async function exportData() {
            try {
                const response = await fetch('/api/export?format=json');
                const data = await response.json();
                
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `crypto_predictor_export_${new Date().toISOString().slice(0, 19)}.json`;
                a.click();
                URL.revokeObjectURL(url);
                
                showMessage('✅ Daten exportiert', 'success');
            } catch (error) {
                showMessage('❌ Fehler beim Export: ' + error.message, 'error');
            }
        }
        
        // Daten laden
        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                if (data.status === 'success') {
                    updateSystemStatus(data.data.system_health);
                    updatePredictions(data.data.predictions);
                    updateMetrics(data.data.system_health.system_metrics);
                    updateLastUpdate(data.data.last_update);
                }
            } catch (error) {
                console.error('Fehler beim Laden des Status:', error);
                updateSystemStatus({ status: 'error' });
            }
        }
        
        async function loadLogs() {
            try {
                const response = await fetch('/api/logs?lines=20');
                const data = await response.json();
                
                if (data.status === 'success') {
                    updateLogs(data.logs);
                }
            } catch (error) {
                console.error('Fehler beim Laden der Logs:', error);
            }
        }
        
        // UI-Updates
        function updateSystemStatus(health) {
            const statusElement = document.getElementById('systemStatus');
            const status = health.status || 'unknown';
            
            statusElement.className = 'status-indicator status-' + status;
            
            const statusTexts = {
                'healthy': '✅ System läuft',
                'warning': '⚠️ Warnung',
                'degraded': '🔶 Beeinträchtigt',
                'error': '❌ Fehler',
                'stopped': '⏹️ Gestoppt'
            };
            
            statusElement.textContent = statusTexts[status] || '❓ Unbekannt';
        }
        
        function updatePredictions(predictions) {
            const container = document.getElementById('predictionsContainer');
            
            if (!predictions || Object.keys(predictions).length === 0) {
                container.innerHTML = '<div class="loading">Keine Vorhersagen verfügbar</div>';
                return;
            }
            
            let html = '';
            
            for (const [symbol, pred] of Object.entries(predictions)) {
                const changeClass = pred.predicted_change >= 0 ? 'change-positive' : 'change-negative';
                const changeSymbol = pred.predicted_change >= 0 ? '+' : '';
                
                html += `
                    <div class="prediction-item">
                        <div class="prediction-symbol">${symbol}</div>
                        <div class="prediction-price">$${pred.current_price.toLocaleString()}</div>
                        <div class="prediction-change ${changeClass}">
                            ${changeSymbol}${pred.predicted_change.toFixed(2)}% 
                            → $${pred.target_price.toLocaleString()}
                        </div>
                        <div>Konfidenz: ${(pred.confidence * 100).toFixed(1)}%</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${pred.confidence * 100}%"></div>
                        </div>
                        <small>Datenqualität: ${(pred.data_quality * 100).toFixed(1)}%</small>
                    </div>
                `;
            }
            
            container.innerHTML = html;
        }
        
        function updateMetrics(metrics) {
            const container = document.getElementById('metricsContainer');
            
            if (!metrics) {
                container.innerHTML = '<div class="loading">Keine Metriken verfügbar</div>';
                return;
            }
            
            const html = `
                <div class="metric-item">
                    <span>CPU-Nutzung:</span>
                    <span class="metric-value">${metrics.cpu_percent?.toFixed(1) || 'N/A'}%</span>
                </div>
                <div class="metric-item">
                    <span>Speicher-Nutzung:</span>
                    <span class="metric-value">${metrics.memory_percent?.toFixed(1) || 'N/A'}%</span>
                </div>
                <div class="metric-item">
                    <span>Verfügbarer Speicher:</span>
                    <span class="metric-value">${metrics.memory_available_mb?.toFixed(0) || 'N/A'} MB</span>
                </div>
                <div class="metric-item">
                    <span>Festplatte:</span>
                    <span class="metric-value">${metrics.disk_percent?.toFixed(1) || 'N/A'}%</span>
                </div>
                <div class="metric-item">
                    <span>Netzwerk:</span>
                    <span class="metric-value">${metrics.network_connectivity ? '✅ OK' : '❌ Fehler'}</span>
                </div>
            `;
            
            container.innerHTML = html;
        }
        
        function updateLogs(logs) {
            const container = document.getElementById('logsContainer');
            
            if (!logs || logs.length === 0) {
                container.innerHTML = '<div class="loading">Keine Logs verfügbar</div>';
                return;
            }
            
            let html = '';
            
            for (const log of logs.slice(0, 10)) {
                const levelClass = 'log-' + log.level.toLowerCase();
                const timestamp = new Date(log.timestamp).toLocaleString();
                
                html += `
                    <div class="log-entry ${levelClass}">
                        <strong>[${timestamp}] ${log.level}:</strong> ${log.message}
                    </div>
                `;
            }
            
            container.innerHTML = html;
        }
        
        function updateLastUpdate(lastUpdate) {
            const element = document.getElementById('lastUpdate');
            
            if (lastUpdate) {
                const date = new Date(lastUpdate);
                element.textContent = date.toLocaleString();
            } else {
                element.textContent = 'Nie';
            }
        }
        
        // Hilfsfunktionen
        function showMessage(message, type = 'info') {
            // Einfache Alert-Implementierung
            alert(message);
        }
        
        function showLoading(message) {
            console.log('Loading:', message);
        }
    </script>
</body>
</html>
    '''
    return html_content

# Template-Verzeichnis erstellen
def create_template_directory():
    """Erstellt Template-Verzeichnis falls nicht vorhanden"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)

# Call this function before running the app
def setup_templates():
    """Setup templates directory and files"""
    create_template_directory()
    
    # Dashboard-Template schreiben
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    template_path = os.path.join(template_dir, 'dashboard.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_template().replace('/templates/dashboard.html', ''))

setup_templates()

def main():
    """Hauptfunktion zum Starten des Dashboards"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Predictor Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host-Adresse')
    parser.add_argument('--port', type=int, default=8080, help='Port-Nummer')
    parser.add_argument('--debug', action='store_true', help='Debug-Modus')
    parser.add_argument('--auto-start', action='store_true', help='System automatisch starten')
    
    args = parser.parse_args()
    
    # System automatisch starten falls gewünscht
    if args.auto_start:
        logger.info("Starte System automatisch...")
        success, message = dashboard_manager.initialize_system()
        if success:
            dashboard_manager.start_auto_refresh()
            logger.info(f"System gestartet: {message}")
        else:
            logger.error(f"Fehler beim automatischen Start: {message}")
    
    logger.info(f"Starte Dashboard auf http://{args.host}:{args.port}")
    
    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("Dashboard beendet")
        dashboard_manager.stop_system()

if __name__ == '__main__':
    main()

