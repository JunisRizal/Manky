# Repository klonen oder Dateien herunterladen
git clone <repository-url> crypto-predictor
cd crypto-predictor

# Automatische Installation
chmod +x install.sh
./install.sh

# System starten
./start.sh

# Dashboard öffnen
open http://localhost:8080


Erste Vorhersage

from minimal_data_predictor import MinimalDataPredictionSystem

# System initialisieren
system = MinimalDataPredictionSystem(['BTC-USD', 'ETH-USD'])

# Vorhersagen machen
predictions = system.make_predictions()

# Ergebnisse anzeigen
for symbol, pred in predictions.items():
    print(f"{symbol}: {pred.predicted_change*100:+.2f}% (Konfidenz: {pred.confidence:.1%})")



# Installationsskript herunterladen empfohlen


curl -O https://raw.githubusercontent.com/your-repo/crypto-predictor/main/install.sh

# Ausführbar machen und installieren
chmod +x install.sh
./install.sh

# Installation überprüfen
./crypto-predictor --version



Manuelle Installation

# Python 3.11+ prüfen
python3 --version

# Virtuelle Umgebung erstellen
python3 -m venv crypto_predictor_env
source crypto_predictor_env/bin/activate  # Linux/Mac
# crypto_predictor_env\Scripts\activate  # Windows



Schritt 3: Systemdateien kopieren

# Arbeitsverzeichnis erstellen
mkdir -p /opt/crypto-predictor
cd /opt/crypto-predictor

# Hauptdateien kopieren
cp /path/to/minimal_data_predictor.py .
cp /path/to/flexible_production_model.py .
cp /path/to/dashboard.py .
cp /path/to/config.json .

# Ausführungsrechte setzen
chmod +x *.py



Schritt 4: Systemservice einrichten (Linux)


# Service-Datei erstellen
sudo cp crypto-predictor.service /etc/systemd/system/

# Service aktivieren
sudo systemctl daemon-reload
sudo systemctl enable crypto-predictor
sudo systemctl start crypto-predictor

# Status prüfen
sudo systemctl status crypto-predictor



# Docker Image bauen

docker build -t crypto-predictor .

# Container starten
docker run -d \
  --name crypto-predictor \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  crypto-predictor

# Logs anzeigen
docker logs -f crypto-predictor


# Mit Docker Compose starten
docker-compose up -d

# Status prüfen
docker-compose ps

# Logs anzeigen
docker-compose logs -f


Haupt-Konfigurationsdatei

{
  "system": {
    "log_level": "INFO",
    "data_directory": "./data",
    "cache_directory": "./cache"
  },
  "api": {
    "rate_limit_interval": 2.0,
    "cache_duration": 300,
    "timeout": 30,
    "retry_attempts": 3
  },
  "prediction": {
    "min_confidence": 0.3,
    "min_data_quality": 0.4,
    "supported_symbols": [
      "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"
    ]
  },
  "dashboard": {
    "host": "0.0.0.0",
    "port": 8080,
    "debug": false,
    "auto_refresh": 60
  }
}


Umgebungsvariablen


# .env Datei erstellen
cat > .env << EOF
CRYPTO_PREDICTOR_CONFIG=./config.json
CRYPTO_PREDICTOR_LOG_LEVEL=INFO
CRYPTO_PREDICTOR_DATA_DIR=./data
CRYPTO_PREDICTOR_CACHE_DIR=./cache
CRYPTO_PREDICTOR_PORT=8080
EOF



Erweiterte Konfiguration

# Konfiguration bearbeiten
nano config.json

# Konfiguration validieren
python3 -c "
import json
with open('config.json') as f:
    config = json.load(f)
print('✅ Konfiguration ist gültig')
print(f'Unterstützte Symbole: {len(config[\"prediction\"][\"supported_symbols\"])}')
"

# Konfiguration neu laden (ohne Neustart)
curl -X POST http://localhost:8080/api/reload-config


Dashboard

# Dashboard starten
python3 dashboard.py

# Oder als Service
systemctl start crypto-predictor-dashboard

# Dashboard öffnen
open http://localhost:8080



Dashboard-Features

Hauptübersicht

•
Live-Vorhersagen: Aktuelle Prognosen für alle Assets

•
Konfidenz-Anzeige: Wissenschaftlich fundierte Unsicherheitsschätzungen

•
Performance-Metriken: Genauigkeit und Fehlerstatistiken

•
System-Status: API-Verbindung und Datenqualität

Kontrollfunktionen

•
Start/Stop: System-Komponenten steuern

•
Konfiguration: Parameter in Echtzeit anpassen

•
Logs: Live-Monitoring der Systemaktivität

•
Export: Daten und Vorhersagen exportieren

Datenvisualisierung

•
Preis-Charts: Historische Kurse mit Vorhersagen

•
Konfidenz-Trends: Entwicklung der Modell-Sicherheit

•
Feature-Wichtigkeit: Einfluss verschiedener Indikatoren

•
Performance-Verlauf: Langzeit-Genauigkeitsanalyse


Kommando Kontrolle


# System-Status prüfen
./crypto-predictor status

# Vorhersagen abrufen
./crypto-predictor predict --symbols BTC-USD,ETH-USD

# Konfiguration anzeigen
./crypto-predictor config show

# Logs anzeigen
./crypto-predictor logs --tail 100

# System neu starten
./crypto-predictor restart

# Backup erstellen
./crypto-predictor backup

# Health-Check
./crypto-predictor health



API Kontrolle

# System-Status
curl http://localhost:8080/api/status

# Vorhersagen abrufen
curl http://localhost:8080/api/predictions

# Spezifische Symbole
curl "http://localhost:8080/api/predictions?symbols=BTC-USD,ETH-USD"

# Konfiguration
curl http://localhost:8080/api/config

# Health-Check
curl http://localhost:8080/api/health


 Internetverbindung und Datenabruf

Netzwerk-Anforderungen

Mindestanforderungen

•
Bandbreite: 1 Mbps Download

•
Latenz: < 500ms zu Yahoo Finance

•
Verfügbarkeit: 95%+ Uptime

•
Ports: Ausgehend 443 (HTTPS)

Empfohlene Konfiguration

•
Bandbreite: 5+ Mbps Download

•
Latenz: < 200ms

•
Verfügbarkeit: 99%+ Uptime

•
Redundanz: Backup-Internetverbindung

Datenquellen-Setup



# API-Konnektivität testen

python3 -c "
from test_crypto_apis import test_crypto_symbols
successful, failed = test_crypto_symbols()
print(f'✅ Erfolgreiche Verbindungen: {len(successful)}')
print(f'❌ Fehlgeschlagene Verbindungen: {len(failed)}')
"


Netzwerkdiagnose

# Internetverbindung testen
ping -c 4 8.8.8.8

# Yahoo Finance erreichbar?
curl -I https://query1.finance.yahoo.com

# DNS-Auflösung testen
nslookup query1.finance.yahoo.com

# Firewall-Regeln prüfen
sudo ufw status



Proxy Konti

# HTTP-Proxy setzen
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# In config.json
{
  "api": {
    "proxy": {
      "http": "http://proxy.company.com:8080",
      "https": "http://proxy.company.com:8080"
    }
  }
}


Offline modus

# Offline-Modus aktivieren (verwendet Cache)
./crypto-predictor --offline

# Cache-Status prüfen
./crypto-predictor cache status

# Cache manuell aktualisieren (wenn online)
./crypto-predictor cache update




# Datenqualität überwachen


from minimal_data_predictor import MinimalDataPredictionSystem

system = MinimalDataPredictionSystem(['BTC-USD'])
current_data, historical = system.data_provider.get_market_data_with_history('BTC-USD')

if current_data:
    print(f"✅ Aktuelle Daten verfügbar: ${current_data.price:,.2f}")
    print(f"📊 Historische Datenpunkte: {len(historical)}")
else:
    print("❌ Keine aktuellen Daten verfügbar")




System initialisieren/ Verwendung

from minimal_data_predictor import MinimalDataPredictionSystem

# Symbole definieren
symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']

# System erstellen
system = MinimalDataPredictionSystem(symbols)



Vorhersagen machen


# Vorhersagen für alle Symbole
predictions = system.make_predictions()

# Ergebnisse verarbeiten
for symbol, prediction in predictions.items():
    print(f"\n{symbol}:")
    print(f"  Aktueller Preis: ${prediction.current_price:,.2f}")
    print(f"  Vorhersage: {prediction.predicted_change*100:+.2f}%")
    print(f"  Zielpreis: ${prediction.target_price:,.2f}")
    print(f"  Konfidenz: {prediction.confidence:.1%}")
    print(f"  Unsicherheit: {prediction.uncertainty:.3f}")





# Detaillierter Status


status = system.get_system_summary()
print(f"Aktive Prädiktoren: {status['active_predictors']}/{status['total_symbols']}")

# Einzelne Modelle prüfen
for symbol, predictor_info in status['predictors'].items():
    print(f"{symbol}: {predictor_info['total_predictions']} Vorhersagen")



Kontinuierliche Überwachung

import time
from datetime import datetime

def continuous_monitoring(interval_minutes=15):
    system = MinimalDataPredictionSystem(['BTC-USD', 'ETH-USD'])
    
    while True:
        try:
            timestamp = datetime.now()
            print(f"\n=== {timestamp.strftime('%Y-%m-%d %H:%M:%S')} ===")
            
            predictions = system.make_predictions()
            
            for symbol, pred in predictions.items():
                print(f"{symbol}: {pred.predicted_change*100:+.2f}% "
                      f"(Konfidenz: {pred.confidence:.1%})")
            
            time.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            print("\nÜberwachung beendet.")
            break
        except Exception as e:
            print(f"Fehler: {e}")
            time.sleep(60)

# Starten
continuous_monitoring()




Batch-Verarbeitung


import pandas as pd

def batch_analysis(symbols, output_file='analysis.csv'):
    system = MinimalDataPredictionSystem(symbols)
    results = []
    
    for symbol in symbols:
        try:
            predictions = system.make_predictions()
            if symbol in predictions:
                pred = predictions[symbol]
                results.append({
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'current_price': pred.current_price,
                    'predicted_change': pred.predicted_change,
                    'confidence': pred.confidence,
                    'data_quality': pred.data_quality
                })
        except Exception as e:
            print(f"Fehler bei {symbol}: {e}")
    
    # Als CSV speichern
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Ergebnisse gespeichert: {output_file}")
    
    return df

# Verwendung
symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD']
df = batch_analysis(symbols)



Rest API


from api_server import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




API Calls


# Alle Vorhersagen
curl http://localhost:5000/api/predictions

# Spezifische Symbole
curl "http://localhost:5000/api/predictions?symbols=BTC-USD,ETH-USD"

# System-Status
curl http://localhost:5000/api/status

# Health-Check
curl http://localhost:5000/api/health




Python API-Client


import requests

class CryptoPredictorClient:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
    
    def get_predictions(self, symbols=None):
        url = f"{self.base_url}/api/predictions"
        if symbols:
            url += f"?symbols={','.join(symbols)}"
        
        response = requests.get(url)
        return response.json()
    
    def get_status(self):
        response = requests.get(f"{self.base_url}/api/status")
        return response.json()

# Verwendung
client = CryptoPredictorClient()
predictions = client.get_predictions(['BTC-USD', 'ETH-USD'])



Häufige Probleme

Problem: "Keine Internetverbindung"

Bash


# Lösung 1: Netzwerk prüfen
ping google.com

# Lösung 2: DNS prüfen
nslookup query1.finance.yahoo.com

# Lösung 3: Firewall prüfen
sudo ufw status

# Lösung 4: Proxy konfigurieren
export HTTP_PROXY=http://proxy:8080


Problem: "API-Rate-Limit erreicht"

Bash


# Lösung: Rate-Limit in config.json erhöhen
{
  "api": {
    "rate_limit_interval": 5.0  // Von 2.0 auf 5.0 erhöhen
  }
}


Problem: "Unzureichende Datenqualität"

Python


# Diagnose: Datenqualität prüfen
system = MinimalDataPredictionSystem(['BTC-USD'])
current_data, historical = system.data_provider.get_market_data_with_history('BTC-USD')

print(f"Aktuelle Daten: {current_data is not None}")
print(f"Historische Datenpunkte: {len(historical)}")

# Lösung: Mindestqualität reduzieren
system = MinimalDataPredictionSystem(['BTC-USD'], min_data_quality=0.3)


Problem: "Dashboard lädt nicht"

Bash


# Lösung 1: Port prüfen
netstat -tulpn | grep 8080

# Lösung 2: Dashboard neu starten
pkill -f dashboard.py
python3 dashboard.py

# Lösung 3: Anderen Port verwenden
python3 dashboard.py --port 8081


Diagnose-Tools

System-Diagnose

Bash


#!/bin/bash
# diagnose.sh

echo "=== Crypto Predictor Diagnose ==="

# Python-Version
echo "Python Version:"
python3 --version

# Abhängigkeiten
echo -e "\nAbhängigkeiten:"
pip list | grep -E "(numpy|pandas|requests|flask)"

# Internetverbindung
echo -e "\nInternetverbindung:"
ping -c 2 8.8.8.8 > /dev/null && echo "✅ Internet OK" || echo "❌ Kein Internet"

# API-Erreichbarkeit
echo -e "\nAPI-Erreichbarkeit:"
curl -s -I https://query1.finance.yahoo.com > /dev/null && echo "✅ Yahoo Finance OK" || echo "❌ Yahoo Finance nicht erreichbar"

# Speicherplatz
echo -e "\nSpeicherplatz:"
df -h . | tail -1

# Arbeitsspeicher
echo -e "\nArbeitsspeicher:"
free -h | head -2

# Prozesse
echo -e "\nLaufende Prozesse:"
ps aux | grep -E "(python.*crypto|dashboard)" | grep -v grep


Log-Analyse

Bash


# Aktuelle Logs anzeigen
tail -f /var/log/crypto-predictor/system.log

# Fehler suchen
grep -i error /var/log/crypto-predictor/system.log | tail -10

# Performance-Probleme
grep -i "slow\|timeout\|failed" /var/log/crypto-predictor/system.log

# Log-Rotation prüfen
ls -la /var/log/crypto-predictor/


Performance-Optimierung

Memory-Optimierung

Python


# Memory-Usage überwachen
import psutil

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
    
    # System-Memory
    system_memory = psutil.virtual_memory()
    print(f"System: {system_memory.percent:.1f}% verwendet")

# Cache-Optimierung
def optimize_cache():
    # Cache-Größe begrenzen
    max_cache_entries = 100
    
    if len(data_provider.data_cache) > max_cache_entries:
        # Älteste Einträge entfernen
        sorted_cache = sorted(data_provider.data_cache.items(), 
                            key=lambda x: x[1][0])  # Nach Timestamp sortieren
        
        for key, _ in sorted_cache[:-max_cache_entries]:
            del data_provider.data_cache[key]


API-Optimierung

Python


# Batch-Requests implementieren
def batch_api_calls(symbols, batch_size=3):
    results = {}
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        
        # Parallele Requests (falls API unterstützt)
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(get_market_data, symbol): symbol 
                      for symbol in batch}
            
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    print(f"Fehler bei {symbol}: {e}")
        
        # Rate-Limiting zwischen Batches
        time.sleep(2.0)
    
    return results






📚 API-Referenz

REST API Endpoints

GET /api/health

Systemgesundheit prüfen

Response:

JSON


{
  "status": "healthy",
  "timestamp": 1690123456.789,
  "checks": {
    "api_connectivity": true,
    "system_integrity": true,
    "prediction_quality": true
  },
  "metrics": {
    "memory_usage_percent": 45.2,
    "active_predictors": 3,
    "cache_entries": 15
  }
}


GET /api/predictions

Vorhersagen abrufen

Parameter:

•
symbols (optional): Komma-getrennte Liste von Symbolen

Response:

JSON


{
  "status": "success",
  "predictions": {
    "BTC-USD": {
      "current_price": 117951.70,
      "predicted_change_percent": 0.15,
      "target_price": 118128.47,
      "confidence": 0.852,
      "uncertainty": 0.075,
      "timestamp": 1690123456.789
    }
  },
  "count": 1
}


GET /api/status

System-Status abrufen

Response:

JSON


{
  "status": "success",
  "system": {
    "active_predictors": 3,
    "total_symbols": 8,
    "uptime_seconds": 3600,
    "version": "2.0"
  },
  "predictors": {
    "BTC-USD": {
      "total_predictions": 45,
      "recent_accuracy": 0.623,
      "can_predict": true
    }
  }
}


POST /api/config/reload

Konfiguration neu laden

Response:

JSON


{
  "status": "success",
  "message": "Configuration reloaded successfully"
}


Python API

MinimalDataPredictionSystem

Python


class MinimalDataPredictionSystem:
    def __init__(self, symbols: List[str], min_data_quality: float = 0.4):
        """
        Initialisiert das Vorhersagesystem
        
        Args:
            symbols: Liste der zu überwachenden Symbole
            min_data_quality: Mindest-Datenqualität (0.0-1.0)
        """
    
    def make_predictions(self) -> Dict[str, PredictionResult]:
        """
        Macht Vorhersagen für alle Symbole
        
        Returns:
            Dictionary mit Vorhersageergebnissen
        """
    
    def get_system_summary(self) -> Dict:
        """
        Gibt System-Zusammenfassung zurück
        
        Returns:
            Dictionary mit System-Metriken
        """


PredictionResult

Python


@dataclass
class PredictionResult:
    symbol: str                    # Symbol (z.B. "BTC-USD")
    timestamp: float              # Unix-Timestamp
    current_price: float          # Aktueller Preis in USD
    predicted_change: float       # Vorhergesagte Änderung (-1.0 bis 1.0)
    target_price: float          # Zielpreis in USD
    confidence: float            # Konfidenz (0.0-1.0)
    uncertainty: float           # Unsicherheit (0.0-1.0)
    method: str                  # Verwendete Methode
    features_used: List[str]     # Verwendete Features
    data_quality: float          # Datenqualität (0.0-1.0)






🛠️ Entwicklung

Entwicklungsumgebung einrichten

Bash


# Repository klonen
git clone <repository-url>
cd crypto-predictor

# Entwicklungsumgebung erstellen
python3 -m venv dev_env
source dev_env/bin/activate

# Entwicklungsabhängigkeiten installieren
pip install -r requirements-dev.txt

# Pre-commit hooks installieren
pre-commit install


Code-Qualität

Bash


# Code formatieren
black *.py

# Linting
flake8 *.py

# Type checking
mypy *.py

# Tests ausführen
pytest tests/

# Coverage-Report
pytest --cov=. tests/


Neue Features entwickeln

Bash


# Feature-Branch erstellen
git checkout -b feature/new-predictor

# Entwickeln und testen
python3 -m pytest tests/test_new_feature.py

# Code-Review vorbereiten
black *.py
flake8 *.py
pytest

# Commit und Push
git add .
git commit -m "Add new predictor feature"
git push origin feature/new-predictor


Debugging

Python


# Debug-Modus aktivieren
import logging
logging.basicConfig(level=logging.DEBUG)

# Detaillierte Logs
logger = logging.getLogger(__name__)
logger.debug("Debug-Information hier")

# Performance-Profiling
import cProfile
cProfile.run('system.make_predictions()')

# Memory-Profiling
from memory_profiler import profile

@profile
def test_memory_usage():
    system = MinimalDataPredictionSystem(['BTC-USD'])
    predictions = system.make_predictions()
    return predictions






📄 Lizenz und Haftungsausschluss

Lizenz

Dieses System ist für wissenschaftliche Forschung und private Bildungszwecke bestimmt.

Haftungsausschluss

•
⚠️ Keine Investitionsberatung: Dieses System stellt keine Finanzberatung dar

•
📊 Experimentell: Alle Vorhersagen sind experimenteller Natur

•
🔬 Forschungszweck: Ausschließlich für wissenschaftliche Analyse

•
💰 Risiko: Investitionen sind immer mit Risiken verbunden

Support

•
📧 Issues: GitHub Issues für Bugs und Feature-Requests

•
📖 Dokumentation: Vollständige Dokumentation verfügbar

•
🤝 Community: Diskussionen und Erfahrungsaustausch





Version: 2.0


Letztes Update: 29. Juli 2025


Status: Produktionsreif

