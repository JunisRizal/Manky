#!/bin/bash
# Automatisches Installationsskript für das Kryptowährungs-Vorhersagesystem
# Version: 2.0
# Datum: 29. Juli 2025

set -e  # Beende bei Fehlern

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging-Funktionen
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner anzeigen
show_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                                                              ║"
    echo "║        🚀 Crypto Predictor Installation v2.0 🚀             ║"
    echo "║                                                              ║"
    echo "║     Wissenschaftliches Kryptowährungs-Vorhersagesystem      ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Systemanforderungen prüfen
check_requirements() {
    log_info "Prüfe Systemanforderungen..."
    
    # Betriebssystem
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        log_success "Betriebssystem: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_success "Betriebssystem: macOS"
    else
        log_error "Nicht unterstütztes Betriebssystem: $OSTYPE"
        exit 1
    fi
    
    # Python-Version prüfen
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 11 ]]; then
            log_success "Python $PYTHON_VERSION gefunden"
        else
            log_error "Python 3.11+ erforderlich, gefunden: $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "Python 3 nicht gefunden"
        exit 1
    fi
    
    # pip prüfen
    if command -v pip3 &> /dev/null; then
        log_success "pip3 verfügbar"
    else
        log_error "pip3 nicht gefunden"
        exit 1
    fi
    
    # Internetverbindung prüfen
    if ping -c 1 8.8.8.8 &> /dev/null; then
        log_success "Internetverbindung verfügbar"
    else
        log_warning "Keine Internetverbindung - Installation könnte fehlschlagen"
    fi
    
    # Speicherplatz prüfen (mindestens 1GB)
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    if [[ $AVAILABLE_SPACE -gt 1048576 ]]; then  # 1GB in KB
        log_success "Ausreichend Speicherplatz verfügbar"
    else
        log_warning "Wenig Speicherplatz verfügbar: $(($AVAILABLE_SPACE/1024))MB"
    fi
}

# Installationsverzeichnis erstellen
create_directories() {
    log_info "Erstelle Installationsverzeichnisse..."
    
    # Hauptverzeichnis
    INSTALL_DIR="/opt/crypto-predictor"
    if [[ $EUID -ne 0 ]]; then
        INSTALL_DIR="$HOME/crypto-predictor"
        log_warning "Keine Root-Rechte - installiere in $INSTALL_DIR"
    fi
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$INSTALL_DIR/data"
    mkdir -p "$INSTALL_DIR/logs"
    mkdir -p "$INSTALL_DIR/cache"
    mkdir -p "$INSTALL_DIR/backups"
    mkdir -p "$INSTALL_DIR/config"
    
    log_success "Verzeichnisse erstellt: $INSTALL_DIR"
    
    # Berechtigungen setzen
    chmod 755 "$INSTALL_DIR"
    chmod 755 "$INSTALL_DIR"/*
}

# Python-Umgebung einrichten
setup_python_environment() {
    log_info "Richte Python-Umgebung ein..."
    
    cd "$INSTALL_DIR"
    
    # Virtuelle Umgebung erstellen
    python3 -m venv crypto_predictor_env
    
    # Aktivieren
    source crypto_predictor_env/bin/activate
    
    # pip upgraden
    pip install --upgrade pip
    
    log_success "Virtuelle Umgebung erstellt"
}

# Abhängigkeiten installieren
install_dependencies() {
    log_info "Installiere Python-Abhängigkeiten..."
    
    cd "$INSTALL_DIR"
    source crypto_predictor_env/bin/activate
    
    # Basis-Abhängigkeiten
    pip install numpy pandas requests flask psutil
    
    # Optionale Abhängigkeiten
    pip install matplotlib plotly || log_warning "Optionale Visualisierungs-Pakete konnten nicht installiert werden"
    
    # Entwicklungs-Abhängigkeiten (optional)
    pip install pytest black flake8 || log_warning "Entwicklungs-Tools konnten nicht installiert werden"
    
    log_success "Abhängigkeiten installiert"
}

# Systemdateien kopieren
copy_system_files() {
    log_info "Kopiere Systemdateien..."
    
    cd "$INSTALL_DIR"
    
    # Hauptdateien (falls im aktuellen Verzeichnis vorhanden)
    for file in minimal_data_predictor.py dashboard.py flexible_production_model.py real_data_integration.py; do
        if [[ -f "../$file" ]]; then
            cp "../$file" .
            chmod +x "$file"
            log_success "Kopiert: $file"
        elif [[ -f "./$file" ]]; then
            chmod +x "$file"
            log_success "Gefunden: $file"
        else
            log_warning "Datei nicht gefunden: $file"
        fi
    done
    
    # Konfigurationsdatei erstellen
    create_config_file
    
    # Startskripte erstellen
    create_start_scripts
}

# Konfigurationsdatei erstellen
create_config_file() {
    log_info "Erstelle Konfigurationsdatei..."
    
    cat > config/config.json << EOF
{
  "system": {
    "log_level": "INFO",
    "log_file": "$INSTALL_DIR/logs/system.log",
    "data_directory": "$INSTALL_DIR/data",
    "cache_directory": "$INSTALL_DIR/cache",
    "backup_directory": "$INSTALL_DIR/backups"
  },
  "api": {
    "rate_limit_interval": 2.0,
    "cache_duration": 300,
    "timeout": 30,
    "retry_attempts": 3,
    "retry_delay": 5
  },
  "prediction": {
    "min_confidence": 0.3,
    "min_data_quality": 0.4,
    "max_prediction_age": 3600,
    "supported_symbols": [
      "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD",
      "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD"
    ]
  },
  "dashboard": {
    "host": "0.0.0.0",
    "port": 8080,
    "debug": false,
    "auto_refresh": 60,
    "auto_start": false
  },
  "monitoring": {
    "health_check_interval": 300,
    "performance_log_interval": 900,
    "backup_interval": 3600,
    "alert_thresholds": {
      "api_error_rate": 0.1,
      "prediction_accuracy": 0.4,
      "memory_usage_mb": 100
    }
  }
}
EOF
    
    log_success "Konfigurationsdatei erstellt"
}

# Startskripte erstellen
create_start_scripts() {
    log_info "Erstelle Startskripte..."
    
    # Hauptstartskript
    cat > start.sh << 'EOF'
#!/bin/bash
# Crypto Predictor Startskript

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Virtuelle Umgebung aktivieren
source crypto_predictor_env/bin/activate

# Umgebungsvariablen setzen
export CRYPTO_PREDICTOR_CONFIG="$SCRIPT_DIR/config/config.json"
export CRYPTO_PREDICTOR_DATA_DIR="$SCRIPT_DIR/data"
export CRYPTO_PREDICTOR_LOG_LEVEL="INFO"

echo "🚀 Starte Crypto Predictor..."
echo "📁 Arbeitsverzeichnis: $SCRIPT_DIR"
echo "⚙️ Konfiguration: $CRYPTO_PREDICTOR_CONFIG"

# Dashboard starten
python3 dashboard.py --auto-start --host 0.0.0.0 --port 8080

EOF
    
    # Dashboard-only Skript
    cat > start-dashboard.sh << 'EOF'
#!/bin/bash
# Nur Dashboard starten

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source crypto_predictor_env/bin/activate

export CRYPTO_PREDICTOR_CONFIG="$SCRIPT_DIR/config/config.json"

echo "📊 Starte Dashboard..."
python3 dashboard.py --host 0.0.0.0 --port 8080

EOF
    
    # CLI-Skript
    cat > crypto-predictor << 'EOF'
#!/bin/bash
# Crypto Predictor CLI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source crypto_predictor_env/bin/activate

export CRYPTO_PREDICTOR_CONFIG="$SCRIPT_DIR/config/config.json"

case "$1" in
    "predict")
        python3 -c "
from minimal_data_predictor import MinimalDataPredictionSystem
import sys

symbols = sys.argv[1].split(',') if len(sys.argv) > 1 else ['BTC-USD', 'ETH-USD']
system = MinimalDataPredictionSystem(symbols)
predictions = system.make_predictions()

for symbol, pred in predictions.items():
    print(f'{symbol}: {pred.predicted_change*100:+.2f}% (Konfidenz: {pred.confidence:.1%})')
" "${2:-BTC-USD,ETH-USD}"
        ;;
    "status")
        python3 -c "
from minimal_data_predictor import MinimalDataPredictionSystem
system = MinimalDataPredictionSystem(['BTC-USD'])
status = system.get_system_summary()
print(f'Aktive Prädiktoren: {status[\"active_predictors\"]}/{status[\"total_symbols\"]}')
"
        ;;
    "health")
        curl -s http://localhost:8080/api/health | python3 -m json.tool
        ;;
    "logs")
        tail -n "${2:-50}" logs/system.log
        ;;
    "version")
        echo "Crypto Predictor v2.0"
        ;;
    *)
        echo "Verwendung: $0 {predict|status|health|logs|version}"
        echo ""
        echo "Befehle:"
        echo "  predict [symbols]  - Vorhersagen machen"
        echo "  status            - System-Status anzeigen"
        echo "  health            - Health-Check"
        echo "  logs [lines]      - Logs anzeigen"
        echo "  version           - Version anzeigen"
        ;;
esac

EOF
    
    # Ausführbar machen
    chmod +x start.sh start-dashboard.sh crypto-predictor
    
    log_success "Startskripte erstellt"
}

# Systemservice einrichten (nur Linux mit Root)
setup_systemd_service() {
    if [[ "$OS" == "linux" && $EUID -eq 0 ]]; then
        log_info "Richte systemd-Service ein..."
        
        cat > /etc/systemd/system/crypto-predictor.service << EOF
[Unit]
Description=Crypto Predictor Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=crypto-predictor
Group=crypto-predictor
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$INSTALL_DIR/crypto_predictor_env/bin
Environment=CRYPTO_PREDICTOR_CONFIG=$INSTALL_DIR/config/config.json
ExecStart=$INSTALL_DIR/crypto_predictor_env/bin/python3 $INSTALL_DIR/dashboard.py --auto-start
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
        
        # Benutzer erstellen
        if ! id "crypto-predictor" &>/dev/null; then
            useradd -r -s /bin/false -d "$INSTALL_DIR" crypto-predictor
            log_success "Benutzer crypto-predictor erstellt"
        fi
        
        # Berechtigungen setzen
        chown -R crypto-predictor:crypto-predictor "$INSTALL_DIR"
        
        # Service aktivieren
        systemctl daemon-reload
        systemctl enable crypto-predictor
        
        log_success "systemd-Service eingerichtet"
        log_info "Service starten mit: sudo systemctl start crypto-predictor"
    else
        log_info "Überspringe systemd-Service (kein Linux oder keine Root-Rechte)"
    fi
}

# Docker-Dateien erstellen
create_docker_files() {
    log_info "Erstelle Docker-Dateien..."
    
    # Dockerfile
    cat > Dockerfile << 'EOF'
FROM python:3.11-slim

# System-Abhängigkeiten
RUN apt-get update && apt-get install -y \
    curl \
    ping \
    && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /app

# Python-Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendungscode
COPY . .

# Benutzer erstellen
RUN useradd -r -s /bin/false crypto-predictor
RUN chown -R crypto-predictor:crypto-predictor /app
USER crypto-predictor

# Ports
EXPOSE 8080

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Startup
CMD ["python3", "dashboard.py", "--auto-start", "--host", "0.0.0.0", "--port", "8080"]
EOF
    
    # requirements.txt
    cat > requirements.txt << 'EOF'
numpy>=1.24.0
pandas>=2.0.0
requests>=2.28.0
flask>=2.3.0
psutil>=5.9.0
matplotlib>=3.6.0
plotly>=5.14.0
EOF
    
    # docker-compose.yml
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  crypto-predictor:
    build: .
    container_name: crypto-predictor
    restart: unless-stopped
    environment:
      - CRYPTO_PREDICTOR_LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
      - ./backups:/app/backups
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  crypto_data:
  crypto_logs:
  crypto_config:
  crypto_backups:
EOF
    
    log_success "Docker-Dateien erstellt"
}

# Installation testen
test_installation() {
    log_info "Teste Installation..."
    
    cd "$INSTALL_DIR"
    source crypto_predictor_env/bin/activate
    
    # Python-Import testen
    python3 -c "
import numpy, pandas, requests, flask
print('✅ Alle Abhängigkeiten importiert')
"
    
    # Systemdateien prüfen
    if [[ -f "minimal_data_predictor.py" ]]; then
        python3 -c "
from minimal_data_predictor import MinimalDataPredictionSystem
print('✅ Hauptmodul importiert')
"
    else
        log_warning "Hauptmodul nicht gefunden - manuell kopieren erforderlich"
    fi
    
    # Konfiguration testen
    python3 -c "
import json
with open('config/config.json') as f:
    config = json.load(f)
print('✅ Konfiguration gültig')
"
    
    log_success "Installation erfolgreich getestet"
}

# Abschlussinformationen anzeigen
show_completion_info() {
    log_success "Installation abgeschlossen!"
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    Installation erfolgreich!                ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}📁 Installationsverzeichnis:${NC} $INSTALL_DIR"
    echo -e "${BLUE}🔧 Konfiguration:${NC} $INSTALL_DIR/config/config.json"
    echo -e "${BLUE}📊 Dashboard:${NC} http://localhost:8080"
    echo ""
    echo -e "${YELLOW}🚀 System starten:${NC}"
    echo "   cd $INSTALL_DIR"
    echo "   ./start.sh"
    echo ""
    echo -e "${YELLOW}📊 Nur Dashboard starten:${NC}"
    echo "   cd $INSTALL_DIR"
    echo "   ./start-dashboard.sh"
    echo ""
    echo -e "${YELLOW}⚙️ CLI verwenden:${NC}"
    echo "   cd $INSTALL_DIR"
    echo "   ./crypto-predictor predict"
    echo "   ./crypto-predictor status"
    echo ""
    
    if [[ "$OS" == "linux" && $EUID -eq 0 ]]; then
        echo -e "${YELLOW}🔧 systemd-Service:${NC}"
        echo "   sudo systemctl start crypto-predictor"
        echo "   sudo systemctl status crypto-predictor"
        echo ""
    fi
    
    echo -e "${YELLOW}🐳 Docker verwenden:${NC}"
    echo "   cd $INSTALL_DIR"
    echo "   docker-compose up -d"
    echo ""
    echo -e "${BLUE}📖 Weitere Informationen:${NC} README.md"
    echo -e "${BLUE}🆘 Support:${NC} GitHub Issues"
    echo ""
    echo -e "${RED}⚠️  WICHTIG:${NC} Dieses System ist nur für wissenschaftliche Forschung bestimmt!"
    echo -e "${RED}   Keine Investitionsberatung - alle Vorhersagen sind experimentell!${NC}"
}

# Hauptfunktion
main() {
    show_banner
    
    # Kommandozeilenargumente verarbeiten
    SKIP_TESTS=false
    INSTALL_DOCKER=true
    INSTALL_SERVICE=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --no-docker)
                INSTALL_DOCKER=false
                shift
                ;;
            --no-service)
                INSTALL_SERVICE=false
                shift
                ;;
            --help)
                echo "Verwendung: $0 [Optionen]"
                echo ""
                echo "Optionen:"
                echo "  --skip-tests    Überspringt Installationstests"
                echo "  --no-docker     Erstellt keine Docker-Dateien"
                echo "  --no-service    Richtet keinen systemd-Service ein"
                echo "  --help          Zeigt diese Hilfe an"
                exit 0
                ;;
            *)
                log_error "Unbekannte Option: $1"
                exit 1
                ;;
        esac
    done
    
    # Installationsschritte
    check_requirements
    create_directories
    setup_python_environment
    install_dependencies
    copy_system_files
    
    if [[ "$INSTALL_DOCKER" == true ]]; then
        create_docker_files
    fi
    
    if [[ "$INSTALL_SERVICE" == true ]]; then
        setup_systemd_service
    fi
    
    if [[ "$SKIP_TESTS" == false ]]; then
        test_installation
    fi
    
    show_completion_info
}

# Fehlerbehandlung
trap 'log_error "Installation fehlgeschlagen in Zeile $LINENO"' ERR

# Skript ausführen
main "$@"

