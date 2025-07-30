#!/usr/bin/env python3
"""
Wissenschaftliches Vorhersagesystem für minimale Datenmengen
Arbeitet mit realen Daten und macht robuste Vorhersagen auch bei begrenzter Datenverfügbarkeit
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import time
import json
import math

logger = logging.getLogger(__name__)

@dataclass
class RealMarketData:
    """Struktur für reale Marktdaten"""
    symbol: str
    timestamp: float
    price: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    funding_rate: Optional[float] = None
    long_short_ratio: Optional[float] = None

@dataclass
class PredictionResult:
    """Struktur für Vorhersageergebnisse"""
    symbol: str
    timestamp: float
    current_price: float
    predicted_change: float
    target_price: float
    confidence: float
    uncertainty: float
    method: str
    features_used: List[str]
    data_quality: float

class MinimalDataProvider:
    """Datenanbieter optimiert für minimale API-Calls"""
    
    def __init__(self):
        self.api_client = ApiClient()
        self.supported_symbols = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD',
            'AVAX-USD', 'DOT-USD', 'LINK-USD'
        ]
        self.last_request_time = {}
        self.min_request_interval = 2.0  # Längere Intervalle
        self.data_cache = {}
        self.cache_duration = 300  # 5 Minuten Cache
        
    def get_market_data_with_history(self, symbol: str) -> Tuple[Optional[RealMarketData], List[RealMarketData]]:
        """Holt aktuelle und historische Daten in einem API-Call"""
        if symbol not in self.supported_symbols:
            return None, []
        
        # Cache prüfen
        cache_key = f"{symbol}_data"
        if cache_key in self.data_cache:
            cache_time, current_data, historical_data = self.data_cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                logger.debug(f"Cache hit für {symbol}")
                return current_data, historical_data
        
        # Rate Limiting
        self._enforce_rate_limit(symbol)
        
        try:
            # Hole mehr Daten mit einem Call
            response = self.api_client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': '1h',
                'range': '1mo',  # 1 Monat für mehr historische Daten
                'includeAdjustedClose': True
            })
            
            self.last_request_time[symbol] = time.time()
            
            if not self._validate_response(response):
                logger.warning(f"Ungültige Antwort für {symbol}")
                return None, []
            
            result = response['chart']['result'][0]
            meta = result['meta']
            timestamps = result.get('timestamp', [])
            quotes = result.get('indicators', {}).get('quote', [{}])[0]
            
            # Extrahiere alle verfügbaren Datenpunkte
            all_data = []
            for i, timestamp in enumerate(timestamps):
                if i < len(quotes.get('close', [])):
                    price = quotes['close'][i]
                    volume = quotes.get('volume', [None] * len(timestamps))[i]
                    
                    if price is not None and price > 0:
                        all_data.append(RealMarketData(
                            symbol=symbol,
                            timestamp=float(timestamp),
                            price=float(price),
                            volume=float(volume) if volume is not None and volume > 0 else None
                        ))
            
            if not all_data:
                logger.warning(f"Keine gültigen Daten für {symbol}")
                return None, []
            
            # Sortiere nach Timestamp
            all_data.sort(key=lambda x: x.timestamp)
            
            # Aktuellste Daten als current, Rest als historical
            current_data = all_data[-1]
            historical_data = all_data[:-1] if len(all_data) > 1 else []
            
            # Cache speichern
            self.data_cache[cache_key] = (time.time(), current_data, historical_data)
            
            logger.info(f"{symbol}: {len(historical_data)} historische + 1 aktuelle Datenpunkte")
            return current_data, historical_data
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen von Daten für {symbol}: {str(e)}")
            return None, []
    
    def _enforce_rate_limit(self, symbol: str):
        """Rate Limiting"""
        if symbol in self.last_request_time:
            time_since_last = time.time() - self.last_request_time[symbol]
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
    
    def _validate_response(self, response: Dict) -> bool:
        """Validiert API-Antwort"""
        return (response and 'chart' in response and 
                response['chart'].get('result') and 
                len(response['chart']['result']) > 0)

class ScientificMinimalPredictor:
    """Wissenschaftlicher Prädiktor für minimale Datenmengen"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.prediction_history = []
        self.model_parameters = {
            'momentum_weight': 0.3,
            'volatility_weight': 0.2,
            'volume_weight': 0.1,
            'time_weight': 0.1,
            'trend_weight': 0.3
        }
        self.learning_rate = 0.05
        self.min_confidence = 0.3
        
        logger.info(f"ScientificMinimalPredictor für {symbol} initialisiert")
    
    def predict(self, current_data: RealMarketData, 
                historical_data: List[RealMarketData]) -> Optional[PredictionResult]:
        """Macht wissenschaftliche Vorhersage mit minimalen Daten"""
        
        if len(historical_data) < 2:
            logger.debug(f"Nicht genügend historische Daten für {self.symbol}: {len(historical_data)}")
            return None
        
        # Extrahiere Features
        features = self._extract_minimal_features(current_data, historical_data)
        
        if not features:
            logger.debug(f"Feature-Extraktion fehlgeschlagen für {self.symbol}")
            return None
        
        # Berechne Vorhersage
        prediction_components = self._calculate_prediction_components(features)
        
        # Gewichtete Kombination
        predicted_change = sum(
            self.model_parameters[key] * value 
            for key, value in prediction_components.items()
            if key in self.model_parameters
        )
        
        # Unsicherheit und Konfidenz
        uncertainty = self._calculate_uncertainty(features, historical_data)
        confidence = self._calculate_confidence(features, uncertainty)
        
        if confidence < self.min_confidence:
            logger.debug(f"Konfidenz zu niedrig für {self.symbol}: {confidence:.3f}")
            return None
        
        # Zielpreis berechnen
        target_price = current_data.price * (1 + predicted_change)
        
        result = PredictionResult(
            symbol=self.symbol,
            timestamp=current_data.timestamp,
            current_price=current_data.price,
            predicted_change=predicted_change,
            target_price=target_price,
            confidence=confidence,
            uncertainty=uncertainty,
            method="minimal_scientific",
            features_used=list(features.keys()),
            data_quality=self._assess_data_quality(current_data, historical_data)
        )
        
        logger.info(f"{self.symbol}: Vorhersage {predicted_change*100:.2f}% "
                   f"(Konfidenz: {confidence:.3f})")
        
        return result
    
    def _extract_minimal_features(self, current_data: RealMarketData, 
                                 historical_data: List[RealMarketData]) -> Dict[str, float]:
        """Extrahiert Features aus minimalen Daten"""
        features = {}
        
        if len(historical_data) < 2:
            return features
        
        # Preise extrahieren
        prices = [d.price for d in historical_data if d.price is not None]
        prices.append(current_data.price)
        
        if len(prices) < 3:
            return features
        
        # 1. Momentum (kurzfristig)
        if len(prices) >= 3:
            recent_change = (prices[-1] - prices[-2]) / prices[-2]
            features['momentum'] = np.tanh(recent_change * 10)  # Normalisiert
        
        # 2. Trend (mittelfristig)
        if len(prices) >= 5:
            trend_change = (prices[-1] - prices[-5]) / prices[-5]
            features['trend'] = np.tanh(trend_change * 5)
        elif len(prices) >= 3:
            trend_change = (prices[-1] - prices[0]) / prices[0]
            features['trend'] = np.tanh(trend_change * 5)
        
        # 3. Volatilität
        if len(prices) >= 5:
            volatility = np.std(prices[-5:]) / np.mean(prices[-5:])
            features['volatility'] = min(volatility * 10, 1.0)
        elif len(prices) >= 3:
            volatility = np.std(prices) / np.mean(prices)
            features['volatility'] = min(volatility * 10, 1.0)
        
        # 4. Volume-Trend (falls verfügbar)
        volumes = [d.volume for d in historical_data if d.volume is not None]
        if current_data.volume is not None:
            volumes.append(current_data.volume)
        
        if len(volumes) >= 3:
            volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] > 0 else 0
            features['volume_trend'] = np.tanh(volume_change)
        
        # 5. Zeitbasierte Features
        hour = datetime.fromtimestamp(current_data.timestamp).hour
        features['hour_cycle'] = np.sin(2 * np.pi * hour / 24)
        
        day_of_week = datetime.fromtimestamp(current_data.timestamp).weekday()
        features['day_cycle'] = np.sin(2 * np.pi * day_of_week / 7)
        
        # 6. Relative Position (Support/Resistance)
        if len(prices) >= 5:
            max_price = max(prices[-10:]) if len(prices) >= 10 else max(prices)
            min_price = min(prices[-10:]) if len(prices) >= 10 else min(prices)
            
            if max_price > min_price:
                relative_position = (current_data.price - min_price) / (max_price - min_price)
                features['relative_position'] = relative_position * 2 - 1  # [-1, 1]
        
        return features
    
    def _calculate_prediction_components(self, features: Dict[str, float]) -> Dict[str, float]:
        """Berechnet Vorhersage-Komponenten"""
        components = {}
        
        # Momentum-Komponente
        if 'momentum' in features:
            components['momentum_component'] = features['momentum'] * 0.5
        
        # Trend-Komponente
        if 'trend' in features:
            components['trend_component'] = features['trend'] * 0.3
        
        # Volatilitäts-Komponente (konträr)
        if 'volatility' in features:
            # Hohe Volatilität deutet auf Umkehr hin
            components['volatility_component'] = -features['volatility'] * 0.1
        
        # Volume-Komponente
        if 'volume_trend' in features:
            components['volume_component'] = features['volume_trend'] * 0.2
        
        # Zeit-Komponente (schwach)
        if 'hour_cycle' in features and 'day_cycle' in features:
            time_signal = (features['hour_cycle'] + features['day_cycle']) / 2
            components['time_component'] = time_signal * 0.05
        
        # Mean Reversion (bei extremen Positionen)
        if 'relative_position' in features:
            # Extreme Positionen neigen zur Umkehr
            extreme_factor = abs(features['relative_position'])
            if extreme_factor > 0.8:
                components['reversion_component'] = -np.sign(features['relative_position']) * 0.1
        
        return components
    
    def _calculate_uncertainty(self, features: Dict[str, float], 
                              historical_data: List[RealMarketData]) -> float:
        """Berechnet Unsicherheit"""
        base_uncertainty = 0.3  # Basis-Unsicherheit
        
        # Reduziere Unsicherheit mit mehr Features
        feature_factor = max(0.5, 1.0 - len(features) * 0.1)
        
        # Reduziere Unsicherheit mit mehr historischen Daten
        data_factor = max(0.5, 1.0 - len(historical_data) * 0.05)
        
        # Erhöhe Unsicherheit bei hoher Volatilität
        volatility_factor = 1.0
        if 'volatility' in features:
            volatility_factor = 1.0 + features['volatility'] * 0.5
        
        total_uncertainty = base_uncertainty * feature_factor * data_factor * volatility_factor
        return min(total_uncertainty, 0.8)  # Maximal 80% Unsicherheit
    
    def _calculate_confidence(self, features: Dict[str, float], uncertainty: float) -> float:
        """Berechnet Konfidenz"""
        base_confidence = 1.0 - uncertainty
        
        # Bonus für starke Signale
        signal_strength = 0.0
        if 'momentum' in features:
            signal_strength += abs(features['momentum']) * 0.3
        if 'trend' in features:
            signal_strength += abs(features['trend']) * 0.2
        
        # Bonus für Konsistenz zwischen Signalen
        consistency_bonus = 0.0
        if 'momentum' in features and 'trend' in features:
            if np.sign(features['momentum']) == np.sign(features['trend']):
                consistency_bonus = 0.1
        
        confidence = base_confidence + signal_strength + consistency_bonus
        return np.clip(confidence, 0.0, 1.0)
    
    def _assess_data_quality(self, current_data: RealMarketData, 
                           historical_data: List[RealMarketData]) -> float:
        """Bewertet Datenqualität"""
        quality = 0.5  # Basis-Qualität
        
        # Bonus für aktuellen Preis
        if current_data.price is not None and current_data.price > 0:
            quality += 0.2
        
        # Bonus für Volume
        if current_data.volume is not None and current_data.volume > 0:
            quality += 0.1
        
        # Bonus für historische Daten
        valid_historical = sum(1 for d in historical_data if d.price is not None and d.price > 0)
        quality += min(valid_historical * 0.02, 0.2)
        
        return min(quality, 1.0)
    
    def update_model(self, prediction_result: PredictionResult, actual_change: float):
        """Aktualisiert Modell basierend auf tatsächlichen Ergebnissen"""
        if prediction_result is None:
            return
        
        error = abs(prediction_result.predicted_change - actual_change)
        direction_correct = np.sign(prediction_result.predicted_change) == np.sign(actual_change)
        
        # Einfache Parameteranpassung
        if direction_correct:
            # Verstärke erfolgreiche Parameter leicht
            for key in self.model_parameters:
                self.model_parameters[key] *= 1.01
        else:
            # Schwäche fehlerhafte Parameter leicht
            for key in self.model_parameters:
                self.model_parameters[key] *= 0.99
        
        # Normalisiere Gewichte
        total_weight = sum(self.model_parameters.values())
        if total_weight > 0:
            for key in self.model_parameters:
                self.model_parameters[key] /= total_weight
        
        # Speichere Ergebnis
        self.prediction_history.append({
            'timestamp': prediction_result.timestamp,
            'predicted': prediction_result.predicted_change,
            'actual': actual_change,
            'error': error,
            'direction_correct': direction_correct
        })
        
        logger.info(f"Modell für {self.symbol} aktualisiert: Fehler={error:.4f}, "
                   f"Richtung korrekt={direction_correct}")

class MinimalDataPredictionSystem:
    """Vorhersagesystem für minimale Datenmengen"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data_provider = MinimalDataProvider()
        self.predictors = {}
        
        # Initialisiere Prädiktoren
        for symbol in symbols:
            if symbol in self.data_provider.supported_symbols:
                self.predictors[symbol] = ScientificMinimalPredictor(symbol)
                logger.info(f"Prädiktor für {symbol} initialisiert")
    
    def make_predictions(self) -> Dict[str, PredictionResult]:
        """Macht Vorhersagen für alle Symbole"""
        predictions = {}
        
        for symbol in self.predictors.keys():
            try:
                # Hole Daten
                current_data, historical_data = self.data_provider.get_market_data_with_history(symbol)
                
                if current_data is None:
                    logger.warning(f"Keine Daten für {symbol} verfügbar")
                    continue
                
                # Mache Vorhersage
                prediction = self.predictors[symbol].predict(current_data, historical_data)
                
                if prediction is not None:
                    predictions[symbol] = prediction
                else:
                    logger.debug(f"Keine Vorhersage für {symbol} möglich")
                
            except Exception as e:
                logger.error(f"Fehler bei Vorhersage für {symbol}: {str(e)}")
        
        return predictions
    
    def get_system_summary(self) -> Dict:
        """Gibt Systemzusammenfassung zurück"""
        summary = {
            'timestamp': time.time(),
            'total_symbols': len(self.symbols),
            'active_predictors': len(self.predictors),
            'predictors': {}
        }
        
        for symbol, predictor in self.predictors.items():
            summary['predictors'][symbol] = {
                'total_predictions': len(predictor.prediction_history),
                'model_parameters': predictor.model_parameters.copy(),
                'min_confidence': predictor.min_confidence
            }
            
            if predictor.prediction_history:
                recent_predictions = predictor.prediction_history[-10:]
                accuracy = sum(1 for p in recent_predictions if p['direction_correct']) / len(recent_predictions)
                avg_error = np.mean([p['error'] for p in recent_predictions])
                
                summary['predictors'][symbol].update({
                    'recent_accuracy': accuracy,
                    'average_error': avg_error
                })
        
        return summary

def demonstrate_minimal_system():
    """Demonstriert das System für minimale Datenmengen"""
    print("=== Wissenschaftliches System für minimale Datenmengen ===")
    print()
    
    # Initialisiere System
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    system = MinimalDataPredictionSystem(symbols)
    
    print(f"System initialisiert mit {len(system.predictors)} Prädiktoren")
    print()
    
    # Mache Vorhersagen
    print("=== Vorhersagen mit minimalen Daten ===")
    predictions = system.make_predictions()
    
    if predictions:
        for symbol, prediction in predictions.items():
            print(f"\n{symbol}:")
            print(f"  Aktueller Preis: ${prediction.current_price:,.2f}")
            print(f"  Vorhergesagte Änderung: {prediction.predicted_change*100:+.2f}%")
            print(f"  Zielpreis: ${prediction.target_price:,.2f}")
            print(f"  Konfidenz: {prediction.confidence:.3f}")
            print(f"  Unsicherheit: {prediction.uncertainty:.3f}")
            print(f"  Datenqualität: {prediction.data_quality:.3f}")
            print(f"  Methode: {prediction.method}")
            print(f"  Features: {', '.join(prediction.features_used)}")
    else:
        print("Keine Vorhersagen möglich")
    
    # Systemzusammenfassung
    print("\n=== Systemzusammenfassung ===")
    summary = system.get_system_summary()
    print(f"Aktive Prädiktoren: {summary['active_predictors']}/{summary['total_symbols']}")
    
    for symbol, predictor_info in summary['predictors'].items():
        print(f"\n{symbol}:")
        print(f"  Gesamte Vorhersagen: {predictor_info['total_predictions']}")
        if 'recent_accuracy' in predictor_info:
            print(f"  Aktuelle Genauigkeit: {predictor_info['recent_accuracy']:.3f}")
            print(f"  Durchschnittlicher Fehler: {predictor_info['average_error']:.4f}")
    
    # Exportiere Ergebnisse
    if predictions:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/ubuntu/minimal_predictions_{timestamp}.json"
        
        export_data = {
            'timestamp': time.time(),
            'predictions': {symbol: {
                'symbol': pred.symbol,
                'current_price': pred.current_price,
                'predicted_change': pred.predicted_change,
                'target_price': pred.target_price,
                'confidence': pred.confidence,
                'uncertainty': pred.uncertainty,
                'method': pred.method,
                'features_used': pred.features_used,
                'data_quality': pred.data_quality
            } for symbol, pred in predictions.items()},
            'system_summary': summary
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"\nErgebnisse exportiert: {filename}")
    
    print("\n=== Demonstration abgeschlossen ===")

if __name__ == "__main__":
    # Konfiguriere Logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Führe Demonstration aus
    demonstrate_minimal_system()

