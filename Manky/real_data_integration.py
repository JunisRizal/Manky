#!/usr/bin/env python3
"""
Integration realer Marktdaten für das Produktionsmodell
Verwendet Yahoo Finance API für echte Kryptowährungsdaten
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
from production_model import RealMarketData, ProductionCryptoPredictor, DataQualityManager, AdaptiveFeatureExtractor

logger = logging.getLogger(__name__)

class RealDataProvider:
    """Liefert reale Marktdaten von Yahoo Finance API"""
    
    def __init__(self):
        self.api_client = ApiClient()
        self.supported_symbols = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD',
            'AVAX-USD', 'DOT-USD', 'LINK-USD'
        ]
        self.last_request_time = {}
        self.min_request_interval = 1.0  # Minimum 1 Sekunde zwischen Requests
        
    def get_current_market_data(self, symbol: str) -> Optional[RealMarketData]:
        """Holt aktuelle Marktdaten für ein Symbol"""
        if symbol not in self.supported_symbols:
            logger.warning(f"Symbol {symbol} nicht unterstützt")
            return None
        
        # Rate Limiting
        if symbol in self.last_request_time:
            time_since_last = time.time() - self.last_request_time[symbol]
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
        
        try:
            response = self.api_client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': '1h',
                'range': '1d',
                'includeAdjustedClose': True
            })
            
            self.last_request_time[symbol] = time.time()
            
            if not response or 'chart' not in response or not response['chart']['result']:
                logger.warning(f"Keine Daten für {symbol} erhalten")
                return None
            
            result = response['chart']['result'][0]
            meta = result['meta']
            
            # Aktueller Preis und Timestamp
            current_price = meta.get('regularMarketPrice')
            current_time = time.time()
            
            # Volume aus den letzten Daten
            timestamps = result.get('timestamp', [])
            quotes = result.get('indicators', {}).get('quote', [{}])[0]
            
            current_volume = None
            if 'volume' in quotes and quotes['volume']:
                # Nehme letztes verfügbares Volume
                volumes = [v for v in quotes['volume'] if v is not None]
                if volumes:
                    current_volume = volumes[-1]
            
            return RealMarketData(
                symbol=symbol,
                timestamp=current_time,
                price=current_price,
                volume=current_volume,
                open_interest=None,  # Nicht verfügbar über Yahoo Finance
                funding_rate=None,   # Nicht verfügbar über Yahoo Finance
                long_short_ratio=None  # Nicht verfügbar über Yahoo Finance
            )
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen von Daten für {symbol}: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> List[RealMarketData]:
        """Holt historische Daten für ein Symbol"""
        if symbol not in self.supported_symbols:
            logger.warning(f"Symbol {symbol} nicht unterstützt")
            return []
        
        # Rate Limiting
        if symbol in self.last_request_time:
            time_since_last = time.time() - self.last_request_time[symbol]
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
        
        try:
            # Bestimme Zeitraum basierend auf gewünschten Tagen
            if days <= 7:
                range_param = f"{days}d"
                interval = '1h'
            elif days <= 30:
                range_param = '1mo'
                interval = '1d'
            else:
                range_param = '3mo'
                interval = '1d'
            
            response = self.api_client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': interval,
                'range': range_param,
                'includeAdjustedClose': True
            })
            
            self.last_request_time[symbol] = time.time()
            
            if not response or 'chart' not in response or not response['chart']['result']:
                logger.warning(f"Keine historischen Daten für {symbol} erhalten")
                return []
            
            result = response['chart']['result'][0]
            timestamps = result.get('timestamp', [])
            quotes = result.get('indicators', {}).get('quote', [{}])[0]
            
            historical_data = []
            
            for i, timestamp in enumerate(timestamps):
                if i < len(quotes.get('close', [])):
                    price = quotes['close'][i]
                    volume = quotes.get('volume', [None] * len(timestamps))[i]
                    
                    if price is not None:  # Nur gültige Preisdaten
                        historical_data.append(RealMarketData(
                            symbol=symbol,
                            timestamp=float(timestamp),
                            price=float(price),
                            volume=float(volume) if volume is not None else None,
                            open_interest=None,
                            funding_rate=None,
                            long_short_ratio=None
                        ))
            
            logger.info(f"Historische Daten für {symbol} geladen: {len(historical_data)} Datenpunkte")
            return historical_data
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen historischer Daten für {symbol}: {str(e)}")
            return []
    
    def get_supported_symbols(self) -> List[str]:
        """Gibt Liste der unterstützten Symbole zurück"""
        return self.supported_symbols.copy()

class RealTimePredictor:
    """Echtzeit-Vorhersagesystem mit realen Daten"""
    
    def __init__(self, symbols: List[str], min_data_quality: float = 0.6):
        self.symbols = symbols
        self.data_provider = RealDataProvider()
        self.models = {}
        self.historical_data = {}
        
        # Initialisiere Modelle für jedes Symbol
        for symbol in symbols:
            if symbol in self.data_provider.get_supported_symbols():
                self.models[symbol] = ProductionCryptoPredictor(
                    symbol=symbol, 
                    min_data_quality=min_data_quality
                )
                self.historical_data[symbol] = []
                logger.info(f"Modell für {symbol} initialisiert")
            else:
                logger.warning(f"Symbol {symbol} nicht unterstützt")
    
    def initialize_with_historical_data(self, days: int = 30):
        """Initialisiert Modelle mit historischen Daten"""
        logger.info(f"Lade historische Daten für {days} Tage...")
        
        for symbol in self.models.keys():
            historical = self.data_provider.get_historical_data(symbol, days)
            if historical:
                self.historical_data[symbol] = historical
                logger.info(f"{symbol}: {len(historical)} historische Datenpunkte geladen")
                
                # Trainiere Modell mit historischen Daten
                self._train_with_historical_data(symbol)
            else:
                logger.warning(f"Keine historischen Daten für {symbol} verfügbar")
    
    def _train_with_historical_data(self, symbol: str):
        """Trainiert Modell mit verfügbaren historischen Daten"""
        if symbol not in self.historical_data or len(self.historical_data[symbol]) < 10:
            logger.warning(f"Nicht genügend historische Daten für {symbol}")
            return
        
        model = self.models[symbol]
        historical = self.historical_data[symbol]
        
        # Trainiere mit historischen Daten (außer den letzten 5 für Validierung)
        training_data = historical[:-5] if len(historical) > 5 else historical
        validation_data = historical[-5:] if len(historical) > 5 else []
        
        logger.info(f"Trainiere {symbol} mit {len(training_data)} Datenpunkten")
        
        for i in range(10, len(training_data)):
            current_data = training_data[i]
            previous_data = training_data[:i]
            
            # Feature-Extraktion
            features = model.feature_extractor.extract_features(
                current_data, previous_data, model.data_quality_manager
            )
            
            if features is None:
                continue
            
            # Berechne tatsächliche Preisänderung (falls nächster Datenpunkt verfügbar)
            if i + 1 < len(training_data):
                next_price = training_data[i + 1].price
                actual_change = (next_price - current_data.price) / current_data.price
                
                # Mache Vorhersage (falls möglich)
                prediction_result = model.predict_with_uncertainty(features)
                
                # Aktualisiere Modell
                model.update_model(features, actual_change, prediction_result)
        
        logger.info(f"Training für {symbol} abgeschlossen")
    
    def make_predictions(self) -> Dict[str, Dict]:
        """Macht Vorhersagen für alle Symbole mit aktuellen Daten"""
        predictions = {}
        
        for symbol in self.models.keys():
            try:
                # Hole aktuelle Marktdaten
                current_data = self.data_provider.get_current_market_data(symbol)
                
                if current_data is None:
                    logger.warning(f"Keine aktuellen Daten für {symbol} verfügbar")
                    continue
                
                # Feature-Extraktion
                model = self.models[symbol]
                features = model.feature_extractor.extract_features(
                    current_data, self.historical_data[symbol], model.data_quality_manager
                )
                
                if features is None:
                    logger.warning(f"Feature-Extraktion für {symbol} fehlgeschlagen")
                    continue
                
                # Vorhersage
                prediction_result = model.predict_with_uncertainty(features)
                
                if prediction_result is not None:
                    predictions[symbol] = {
                        'current_price': current_data.price,
                        'predicted_change': prediction_result['prediction'],
                        'predicted_price': current_data.price * (1 + prediction_result['prediction']),
                        'confidence': prediction_result['confidence'],
                        'uncertainty': prediction_result['uncertainty'],
                        'data_quality': prediction_result['data_quality'],
                        'timestamp': current_data.timestamp,
                        'feature_count': prediction_result['feature_count']
                    }
                    
                    logger.info(f"{symbol}: Vorhersage {prediction_result['prediction']:.4f} "
                              f"(Konfidenz: {prediction_result['confidence']:.3f})")
                else:
                    logger.info(f"{symbol}: Keine Vorhersage möglich (unzureichende Datenqualität)")
                
                # Aktualisiere historische Daten
                self.historical_data[symbol].append(current_data)
                
                # Begrenze historische Daten (behalte nur letzte 1000 Punkte)
                if len(self.historical_data[symbol]) > 1000:
                    self.historical_data[symbol] = self.historical_data[symbol][-1000:]
                
            except Exception as e:
                logger.error(f"Fehler bei Vorhersage für {symbol}: {str(e)}")
        
        return predictions
    
    def get_model_status(self) -> Dict[str, Dict]:
        """Gibt Status aller Modelle zurück"""
        status = {}
        
        for symbol, model in self.models.items():
            model_status = model.get_model_status()
            model_status.update({
                'historical_data_points': len(self.historical_data.get(symbol, [])),
                'last_data_timestamp': self.historical_data[symbol][-1].timestamp if self.historical_data.get(symbol) else None
            })
            status[symbol] = model_status
        
        return status
    
    def update_models_with_actual_data(self, actual_changes: Dict[str, float]):
        """Aktualisiert Modelle mit tatsächlichen Preisänderungen"""
        for symbol, actual_change in actual_changes.items():
            if symbol in self.models and len(self.historical_data[symbol]) >= 2:
                model = self.models[symbol]
                
                # Verwende vorletzten Datenpunkt für Feature-Extraktion
                previous_data = self.historical_data[symbol][-2]
                historical = self.historical_data[symbol][:-2]
                
                features = model.feature_extractor.extract_features(
                    previous_data, historical, model.data_quality_manager
                )
                
                if features is not None:
                    # Hole letzte Vorhersage (falls vorhanden)
                    last_prediction = None
                    if model.prediction_history:
                        last_prediction = model.prediction_history[-1]
                    
                    # Aktualisiere Modell
                    model.update_model(features, actual_change, last_prediction)
                    logger.info(f"Modell für {symbol} mit tatsächlicher Änderung {actual_change:.4f} aktualisiert")

def demonstrate_real_time_system():
    """Demonstriert das Echtzeit-Vorhersagesystem"""
    print("=== Echtzeit-Vorhersagesystem mit realen Daten ===")
    print()
    
    # Initialisiere System mit ausgewählten Kryptowährungen
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    predictor = RealTimePredictor(symbols, min_data_quality=0.5)
    
    # Lade historische Daten
    print("Lade historische Daten...")
    predictor.initialize_with_historical_data(days=30)
    
    # Zeige Modellstatus
    print("\n=== Modellstatus ===")
    status = predictor.get_model_status()
    for symbol, model_status in status.items():
        print(f"{symbol}:")
        print(f"  Kann Vorhersagen machen: {model_status['can_predict']}")
        print(f"  Historische Datenpunkte: {model_status['historical_data_points']}")
        print(f"  Parameter-Anzahl: {model_status['parameter_count']}")
        if 'average_accuracy' in model_status:
            print(f"  Durchschnittliche Genauigkeit: {model_status['average_accuracy']:.3f}")
        print()
    
    # Mache Vorhersagen
    print("=== Aktuelle Vorhersagen ===")
    predictions = predictor.make_predictions()
    
    if predictions:
        for symbol, pred in predictions.items():
            print(f"{symbol}:")
            print(f"  Aktueller Preis: ${pred['current_price']:,.2f}")
            print(f"  Vorhergesagte Änderung: {pred['predicted_change']:.4f} ({pred['predicted_change']*100:.2f}%)")
            print(f"  Vorhergesagter Preis: ${pred['predicted_price']:,.2f}")
            print(f"  Konfidenz: {pred['confidence']:.3f}")
            print(f"  Unsicherheit: {pred['uncertainty']:.4f}")
            print(f"  Datenqualität: {pred['data_quality']:.3f}")
            print(f"  Features verwendet: {pred['feature_count']}")
            print()
    else:
        print("Keine Vorhersagen möglich - unzureichende Datenqualität oder zu wenig historische Daten")
    
    print("=== Demonstration abgeschlossen ===")

if __name__ == "__main__":
    # Konfiguriere Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Führe Demonstration aus
    demonstrate_real_time_system()

