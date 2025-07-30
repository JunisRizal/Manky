#!/usr/bin/env python3
"""
Integriertes Vorhersagesystem mit flexiblem Modell und realen Daten
Wissenschaftlich robuste Implementierung für Produktionsumgebung
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
import time
import json
from flexible_production_model import (
    RealMarketData, FlexibleCryptoPredictor, 
    FlexibleDataQualityManager, FlexibleFeatureExtractor
)

logger = logging.getLogger(__name__)

class ScientificDataProvider:
    """Wissenschaftlich robuster Datenanbieter"""
    
    def __init__(self):
        self.api_client = ApiClient()
        self.supported_symbols = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD',
            'AVAX-USD', 'DOT-USD', 'LINK-USD'
        ]
        self.last_request_time = {}
        self.min_request_interval = 1.0
        self.data_cache = {}
        
    def get_current_market_data(self, symbol: str) -> Optional[RealMarketData]:
        """Holt aktuelle Marktdaten mit Validierung"""
        if symbol not in self.supported_symbols:
            logger.warning(f"Symbol {symbol} nicht unterstützt")
            return None
        
        # Rate Limiting
        self._enforce_rate_limit(symbol)
        
        try:
            response = self.api_client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': '1h',
                'range': '2d',  # Mehr Daten für bessere Qualität
                'includeAdjustedClose': True
            })
            
            self.last_request_time[symbol] = time.time()
            
            if not self._validate_api_response(response):
                logger.warning(f"Ungültige API-Antwort für {symbol}")
                return None
            
            result = response['chart']['result'][0]
            meta = result['meta']
            
            # Validiere Preisdaten
            current_price = meta.get('regularMarketPrice')
            if not self._validate_price(current_price):
                logger.warning(f"Ungültiger Preis für {symbol}: {current_price}")
                return None
            
            # Extrahiere Volume aus aktuellsten Daten
            timestamps = result.get('timestamp', [])
            quotes = result.get('indicators', {}).get('quote', [{}])[0]
            
            current_volume = self._extract_latest_volume(quotes)
            current_time = time.time()
            
            market_data = RealMarketData(
                symbol=symbol,
                timestamp=current_time,
                price=float(current_price),
                volume=float(current_volume) if current_volume is not None else None,
                open_interest=None,
                funding_rate=None,
                long_short_ratio=None
            )
            
            # Cache für spätere Verwendung
            self.data_cache[symbol] = market_data
            
            logger.debug(f"Marktdaten für {symbol} erfolgreich abgerufen: ${current_price:.2f}")
            return market_data
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen von Daten für {symbol}: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> List[RealMarketData]:
        """Holt validierte historische Daten"""
        if symbol not in self.supported_symbols:
            return []
        
        self._enforce_rate_limit(symbol)
        
        try:
            # Bestimme optimale Parameter
            if days <= 7:
                range_param = f"{days}d"
                interval = '1h'
            elif days <= 60:
                range_param = '2mo'
                interval = '1d'
            else:
                range_param = '6mo'
                interval = '1d'
            
            response = self.api_client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': interval,
                'range': range_param,
                'includeAdjustedClose': True
            })
            
            self.last_request_time[symbol] = time.time()
            
            if not self._validate_api_response(response):
                return []
            
            result = response['chart']['result'][0]
            timestamps = result.get('timestamp', [])
            quotes = result.get('indicators', {}).get('quote', [{}])[0]
            
            historical_data = []
            
            for i, timestamp in enumerate(timestamps):
                if i < len(quotes.get('close', [])):
                    price = quotes['close'][i]
                    volume = quotes.get('volume', [None] * len(timestamps))[i]
                    
                    if self._validate_price(price):
                        historical_data.append(RealMarketData(
                            symbol=symbol,
                            timestamp=float(timestamp),
                            price=float(price),
                            volume=float(volume) if volume is not None and volume > 0 else None,
                            open_interest=None,
                            funding_rate=None,
                            long_short_ratio=None
                        ))
            
            # Sortiere nach Timestamp
            historical_data.sort(key=lambda x: x.timestamp)
            
            logger.info(f"Historische Daten für {symbol}: {len(historical_data)} validierte Datenpunkte")
            return historical_data
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen historischer Daten für {symbol}: {str(e)}")
            return []
    
    def _enforce_rate_limit(self, symbol: str):
        """Erzwingt Rate Limiting"""
        if symbol in self.last_request_time:
            time_since_last = time.time() - self.last_request_time[symbol]
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                logger.debug(f"Rate limiting: warte {sleep_time:.2f}s für {symbol}")
                time.sleep(sleep_time)
    
    def _validate_api_response(self, response: Dict) -> bool:
        """Validiert API-Antwort"""
        if not response:
            return False
        if 'chart' not in response:
            return False
        if not response['chart'].get('result'):
            return False
        return True
    
    def _validate_price(self, price) -> bool:
        """Validiert Preisdaten"""
        if price is None:
            return False
        try:
            price_float = float(price)
            return price_float > 0 and np.isfinite(price_float)
        except (ValueError, TypeError):
            return False
    
    def _extract_latest_volume(self, quotes: Dict) -> Optional[float]:
        """Extrahiert letztes gültiges Volume"""
        if 'volume' not in quotes or not quotes['volume']:
            return None
        
        # Suche rückwärts nach gültigem Volume
        for volume in reversed(quotes['volume']):
            if volume is not None and volume > 0:
                return volume
        
        return None

class ScientificPredictionSystem:
    """Wissenschaftliches Vorhersagesystem"""
    
    def __init__(self, symbols: List[str], min_data_quality: float = 0.4):
        self.symbols = symbols
        self.data_provider = ScientificDataProvider()
        self.models = {}
        self.historical_data = {}
        self.prediction_log = []
        
        # Wissenschaftliche Parameter
        self.min_historical_days = 7   # Minimum für wissenschaftliche Validität
        self.max_prediction_age = 3600  # Vorhersagen verfallen nach 1 Stunde
        
        # Initialisiere Modelle
        for symbol in symbols:
            if symbol in self.data_provider.supported_symbols:
                self.models[symbol] = FlexibleCryptoPredictor(
                    symbol=symbol, 
                    min_data_quality=min_data_quality
                )
                self.historical_data[symbol] = []
                logger.info(f"Wissenschaftliches Modell für {symbol} initialisiert")
            else:
                logger.warning(f"Symbol {symbol} nicht verfügbar")
    
    def initialize_system(self, historical_days: int = 30) -> Dict[str, bool]:
        """Initialisiert System mit historischen Daten"""
        logger.info(f"Initialisiere System mit {historical_days} Tagen historischer Daten...")
        
        initialization_results = {}
        
        for symbol in self.models.keys():
            try:
                # Lade historische Daten
                historical = self.data_provider.get_historical_data(symbol, historical_days)
                
                if len(historical) < self.min_historical_days:
                    logger.warning(f"Unzureichende historische Daten für {symbol}: "
                                 f"{len(historical)} < {self.min_historical_days}")
                    initialization_results[symbol] = False
                    continue
                
                self.historical_data[symbol] = historical
                logger.info(f"{symbol}: {len(historical)} historische Datenpunkte geladen")
                
                # Trainiere Modell
                training_success = self._train_model_scientifically(symbol)
                initialization_results[symbol] = training_success
                
            except Exception as e:
                logger.error(f"Initialisierung für {symbol} fehlgeschlagen: {str(e)}")
                initialization_results[symbol] = False
        
        successful_models = sum(initialization_results.values())
        logger.info(f"System initialisiert: {successful_models}/{len(self.models)} Modelle erfolgreich")
        
        return initialization_results
    
    def _train_model_scientifically(self, symbol: str) -> bool:
        """Trainiert Modell mit wissenschaftlichen Standards"""
        if symbol not in self.historical_data or len(self.historical_data[symbol]) < 10:
            return False
        
        model = self.models[symbol]
        historical = self.historical_data[symbol]
        
        # Verwende 80% für Training, 20% für Validierung
        split_point = int(len(historical) * 0.8)
        training_data = historical[:split_point]
        validation_data = historical[split_point:]
        
        logger.info(f"Trainiere {symbol}: {len(training_data)} Training, {len(validation_data)} Validierung")
        
        training_count = 0
        
        # Training mit wissenschaftlicher Validierung
        for i in range(5, len(training_data) - 1):  # Mindestens 5 historische Punkte
            current_data = training_data[i]
            previous_data = training_data[:i]
            
            # Feature-Extraktion
            features = model.feature_extractor.extract_features(
                current_data, previous_data, model.data_quality_manager
            )
            
            if features is None:
                continue
            
            # Berechne tatsächliche Änderung
            next_data = training_data[i + 1]
            actual_change = (next_data.price - current_data.price) / current_data.price
            
            # Mache Vorhersage (falls möglich)
            prediction_result = model.predict_with_uncertainty(features)
            
            # Aktualisiere Modell
            model.update_model(features, actual_change, prediction_result)
            training_count += 1
        
        # Validierung
        validation_accuracy = self._validate_model(symbol, validation_data)
        
        logger.info(f"Training für {symbol} abgeschlossen: {training_count} Samples, "
                   f"Validierungsgenauigkeit: {validation_accuracy:.3f}")
        
        return validation_accuracy > 0.3  # Mindestens 30% Genauigkeit
    
    def _validate_model(self, symbol: str, validation_data: List[RealMarketData]) -> float:
        """Validiert Modell wissenschaftlich"""
        if len(validation_data) < 5:
            return 0.0
        
        model = self.models[symbol]
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(len(validation_data) - 1):
            current_data = validation_data[i]
            
            # Verwende alle verfügbaren historischen Daten für Features
            all_historical = self.historical_data[symbol][:-(len(validation_data)-i)]
            
            features = model.feature_extractor.extract_features(
                current_data, all_historical, model.data_quality_manager
            )
            
            if features is None:
                continue
            
            prediction_result = model.predict_with_uncertainty(features)
            
            if prediction_result is not None:
                next_data = validation_data[i + 1]
                actual_change = (next_data.price - current_data.price) / current_data.price
                predicted_change = prediction_result['prediction']
                
                # Richtungsgenauigkeit
                if np.sign(predicted_change) == np.sign(actual_change):
                    correct_predictions += 1
                
                total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def make_scientific_predictions(self) -> Dict[str, Dict]:
        """Macht wissenschaftlich validierte Vorhersagen"""
        predictions = {}
        current_time = time.time()
        
        for symbol in self.models.keys():
            try:
                # Hole aktuelle Daten
                current_data = self.data_provider.get_current_market_data(symbol)
                
                if current_data is None:
                    logger.warning(f"Keine aktuellen Daten für {symbol}")
                    continue
                
                # Feature-Extraktion
                model = self.models[symbol]
                features = model.feature_extractor.extract_features(
                    current_data, self.historical_data[symbol], model.data_quality_manager
                )
                
                if features is None:
                    logger.debug(f"Feature-Extraktion für {symbol} fehlgeschlagen")
                    continue
                
                # Wissenschaftliche Vorhersage
                prediction_result = model.predict_with_uncertainty(features)
                
                if prediction_result is not None:
                    # Berechne Zielpreis
                    predicted_change = prediction_result['prediction']
                    target_price = current_data.price * (1 + predicted_change)
                    
                    # Zeitstempel für Vorhersage
                    prediction_timestamp = current_time
                    
                    prediction_data = {
                        'symbol': symbol,
                        'current_price': current_data.price,
                        'predicted_change_percent': predicted_change * 100,
                        'target_price': target_price,
                        'confidence': prediction_result['confidence'],
                        'uncertainty': prediction_result['uncertainty'],
                        'data_quality': prediction_result['data_quality'],
                        'feature_count': prediction_result['feature_count'],
                        'prediction_timestamp': prediction_timestamp,
                        'expires_at': prediction_timestamp + self.max_prediction_age,
                        'model_status': model.get_model_status()
                    }
                    
                    predictions[symbol] = prediction_data
                    
                    # Logge Vorhersage
                    self.prediction_log.append(prediction_data)
                    
                    logger.info(f"{symbol}: Vorhersage {predicted_change*100:.2f}% "
                              f"(Konfidenz: {prediction_result['confidence']:.3f}, "
                              f"Zielpreis: ${target_price:.2f})")
                else:
                    logger.debug(f"{symbol}: Keine Vorhersage möglich")
                
                # Aktualisiere historische Daten
                self.historical_data[symbol].append(current_data)
                
                # Begrenze historische Daten
                if len(self.historical_data[symbol]) > 1000:
                    self.historical_data[symbol] = self.historical_data[symbol][-1000:]
                
            except Exception as e:
                logger.error(f"Fehler bei Vorhersage für {symbol}: {str(e)}")
        
        return predictions
    
    def get_system_status(self) -> Dict:
        """Gibt umfassenden Systemstatus zurück"""
        status = {
            'timestamp': time.time(),
            'total_models': len(self.models),
            'active_models': 0,
            'total_predictions': len(self.prediction_log),
            'models': {}
        }
        
        for symbol, model in self.models.items():
            model_status = model.get_model_status()
            model_status.update({
                'historical_data_points': len(self.historical_data.get(symbol, [])),
                'last_data_timestamp': self.historical_data[symbol][-1].timestamp if self.historical_data.get(symbol) else None,
                'data_age_hours': (time.time() - self.historical_data[symbol][-1].timestamp) / 3600 if self.historical_data.get(symbol) else None
            })
            
            if model_status['can_predict']:
                status['active_models'] += 1
            
            status['models'][symbol] = model_status
        
        return status
    
    def export_predictions(self, filename: str = None) -> str:
        """Exportiert Vorhersagen als JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/home/ubuntu/predictions_{timestamp}.json"
        
        export_data = {
            'export_timestamp': time.time(),
            'system_status': self.get_system_status(),
            'recent_predictions': self.prediction_log[-50:] if len(self.prediction_log) > 50 else self.prediction_log
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Vorhersagen exportiert nach: {filename}")
        return filename

def demonstrate_scientific_system():
    """Demonstriert das wissenschaftliche Vorhersagesystem"""
    print("=== Wissenschaftliches Kryptowährungs-Vorhersagesystem ===")
    print()
    
    # Initialisiere System
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    system = ScientificPredictionSystem(symbols, min_data_quality=0.4)
    
    # Initialisiere mit historischen Daten
    print("Initialisiere System...")
    init_results = system.initialize_system(historical_days=30)
    
    print("\n=== Initialisierungsergebnisse ===")
    for symbol, success in init_results.items():
        status = "✓ Erfolgreich" if success else "✗ Fehlgeschlagen"
        print(f"{symbol}: {status}")
    
    # Systemstatus
    print("\n=== Systemstatus ===")
    status = system.get_system_status()
    print(f"Aktive Modelle: {status['active_models']}/{status['total_models']}")
    
    for symbol, model_status in status['models'].items():
        if model_status['can_predict']:
            print(f"\n{symbol}:")
            print(f"  Kann Vorhersagen machen: ✓")
            print(f"  Historische Datenpunkte: {model_status['historical_data_points']}")
            print(f"  Durchschnittliche Genauigkeit: {model_status.get('average_accuracy', 'N/A'):.3f}")
            print(f"  Parameter-Anzahl: {model_status['parameter_count']}")
            if model_status.get('data_age_hours'):
                print(f"  Datenalter: {model_status['data_age_hours']:.1f} Stunden")
    
    # Mache Vorhersagen
    print("\n=== Wissenschaftliche Vorhersagen ===")
    predictions = system.make_scientific_predictions()
    
    if predictions:
        for symbol, pred in predictions.items():
            print(f"\n{symbol}:")
            print(f"  Aktueller Preis: ${pred['current_price']:,.2f}")
            print(f"  Vorhergesagte Änderung: {pred['predicted_change_percent']:+.2f}%")
            print(f"  Zielpreis: ${pred['target_price']:,.2f}")
            print(f"  Konfidenz: {pred['confidence']:.3f}")
            print(f"  Unsicherheit: {pred['uncertainty']:.3f}")
            print(f"  Datenqualität: {pred['data_quality']:.3f}")
            print(f"  Features: {pred['feature_count']}")
            
            expires_in = (pred['expires_at'] - time.time()) / 60
            print(f"  Gültig für: {expires_in:.0f} Minuten")
    else:
        print("Keine wissenschaftlich validen Vorhersagen möglich")
    
    # Exportiere Ergebnisse
    if predictions:
        export_file = system.export_predictions()
        print(f"\nErgebnisse exportiert: {export_file}")
    
    print("\n=== Demonstration abgeschlossen ===")

if __name__ == "__main__":
    # Konfiguriere Logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Führe wissenschaftliche Demonstration aus
    demonstrate_scientific_system()

