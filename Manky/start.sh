#!/bin/bash
# Einfaches Startskript für das Crypto Predictor System
# Version: 2.0

set -e

# Farben für Output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starte Crypto Predictor System...${NC}"

# Arbeitsverzeichnis bestimmen
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}📁 Arbeitsverzeichnis: $SCRIPT_DIR${NC}"

# Prüfe ob Python verfügbar ist
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 nicht gefunden${NC}"
    exit 1
fi

# Prüfe ob Hauptdatei existiert
if [[ ! -f "minimal_data_predictor.py" ]]; then
    echo -e "${RED}❌ Hauptdatei minimal_data_predictor.py nicht gefunden${NC}"
    echo "Bitte führen Sie zuerst die Installation aus: ./install.sh"
    exit 1
fi

# Umgebungsvariablen setzen
export CRYPTO_PREDICTOR_CONFIG="$SCRIPT_DIR/config.json"
export CRYPTO_PREDICTOR_DATA_DIR="$SCRIPT_DIR/data"
export CRYPTO_PREDICTOR_LOG_LEVEL="INFO"

# Verzeichnisse erstellen falls nicht vorhanden
mkdir -p data logs cache backups

echo -e "${BLUE}⚙️ Konfiguration: $CRYPTO_PREDICTOR_CONFIG${NC}"

# Prüfe Internetverbindung
if ping -c 1 8.8.8.8 &> /dev/null; then
    echo -e "${GREEN}✅ Internetverbindung verfügbar${NC}"
else
    echo -e "${RED}⚠️ Keine Internetverbindung - System könnte nicht funktionieren${NC}"
fi

# Dashboard mit Auto-Start starten
echo -e "${GREEN}📊 Starte Dashboard mit Auto-Start...${NC}"
echo -e "${BLUE}🌐 Dashboard wird verfügbar sein unter: http://localhost:8080${NC}"
echo -e "${BLUE}🛑 Zum Beenden: Ctrl+C${NC}"
echo ""

# Dashboard starten
python3 dashboard.py --auto-start --host 0.0.0.0 --port 8080

