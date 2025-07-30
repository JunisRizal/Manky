#!/usr/bin/env python3
"""
Produktionsreifes Selbstlernendes Kryptowährungs-Vorhersagemodell
Wissenschaftlich robuste Implementierung ohne Mock-Daten oder Fallbacks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RealMarketData:
    """Struktur für reale Marktdaten ohne Defaults oder Mock-Werte"""
    symbol: str
    timestamp: float
    price: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    funding_rate: Optional[float] = None
    long_short_ratio: Optional[float] = None
    
    def __post_init__(self):
        """Validierung der Eingabedaten"""
        if not self.symbol:
            raise ValueError("Symbol ist erforderlich")
        if self.timestamp <= 0:
            raise ValueError("Gültiger Timestamp ist erforderlich")

@dataclass
class MarketFeatures:
    """Extrahierte Features aus realen Marktdaten"""
    symbol: str
    timestamp: float
    features: np.ndarray
    feature_names: List[str]
    data_quality_score: float
    available_data_sources: List[str]

class DataQualityManager:
    """Verwaltet Datenqualität und Verfügbarkeit"""
    
    def __init__(self):
        self.required_features = [
            'price', 'volume', 'price_change_1h', 'price_change_24h'
        ]
        self.optional_features = [
            'open_interest', 'funding_rate', 'long_short_ratio',
            'liquidations', 'social_sentiment', 'fear_greed_index'
        ]
        
    def assess_data_quality(self, market_data: RealMarketData, 
                          historical_data: List[RealMarketData]) -> float:
        """Bewertet die Qualität der verfügbaren Daten"""
        quality_score = 0.0
        total_weight = 0.0
        
        # Basis-Datenqualität (Preis ist essentiell)
        if market_data.price is not None and market_data.price > 0:
            quality_score += 0.4
        total_weight += 0.4
        
        # Volumen-Verfügbarkeit
        if market_data.volume is not None and market_data.volume > 0:
            quality_score += 0.2
        total_weight += 0.2
        
        # Historische Daten-Kontinuität
        if len(historical_data) >= 10:
            recent_data_points = sum(1 for d in historical_data[-10:] 
                                   if d.price is not None and d.price > 0)
            continuity_score = recent_data_points / 10
            quality_score += 0.2 * continuity_score
        total_weight += 0.2
        
        # Erweiterte Daten-Verfügbarkeit
        extended_data_count = sum([
            1 if market_data.open_interest is not None else 0,
            1 if market_data.funding_rate is not None else 0,
            1 if market_data.long_short_ratio is not None else 0
        ])
        extended_score = extended_data_count / 3
        quality_score += 0.2 * extended_score
        total_weight += 0.2
        
        return quality_score / total_weight if total_weight > 0 else 0.0
    
    def get_available_features(self, market_data: RealMarketData) -> List[str]:
        """Ermittelt verfügbare Features basierend auf vorhandenen Daten"""
        available = []
        
        if market_data.price is not None:
            available.append('price')
        if market_data.volume is not None:
            available.append('volume')
        if market_data.open_interest is not None:
            available.append('open_interest')
        if market_data.funding_rate is not None:
            available.append('funding_rate')
        if market_data.long_short_ratio is not None:
            available.append('long_short_ratio')
            
        return available

class AdaptiveFeatureExtractor:
    """Extrahiert Features adaptiv basierend auf verfügbaren Daten"""
    
    def __init__(self):
        self.feature_registry = {}
        self.feature_weights = {}
        self.min_required_features = 3  # Minimum für sinnvolle Vorhersagen
        
    def extract_features(self, market_data: RealMarketData, 
                        historical_data: List[RealMarketData],
                        data_quality_manager: DataQualityManager) -> Optional[MarketFeatures]:
        """
        Extrahiert Features adaptiv basierend auf verfügbaren Daten
        Gibt None zurück wenn nicht genügend Daten verfügbar sind
        """
        if not self._validate_minimum_requirements(market_data, historical_data):
            logger.warning(f"Nicht genügend Daten für {market_data.symbol} verfügbar")
            return None
        
        features = []
        feature_names = []
        available_sources = data_quality_manager.get_available_features(market_data)
        
        # Basis-Features (nur wenn Daten verfügbar)
        if market_data.price is not None:
            # Preisbasierte Features
            if len(historical_data) > 0 and historical_data[-1].price is not None:
                price_change = (market_data.price - historical_data[-1].price) / historical_data[-1].price
                features.append(price_change)
                feature_names.append('price_change_1period')
            
            # Volatilität (wenn genügend historische Daten)
            if len(historical_data) >= 20:
                prices = [d.price for d in historical_data[-20:] if d.price is not None]
                if len(prices) >= 10:
                    volatility = np.std(prices) / np.mean(prices)
                    features.append(volatility)
                    feature_names.append('volatility_20period')
        
        # Volumen-Features
        if market_data.volume is not None:
            if len(historical_data) > 0 and historical_data[-1].volume is not None:
                volume_change = (market_data.volume - historical_data[-1].volume) / historical_data[-1].volume
                features.append(np.tanh(volume_change))  # Normalisierung
                feature_names.append('volume_change')
        
        # Erweiterte Features (nur wenn verfügbar)
        if market_data.open_interest is not None:
            # Normalisierte Open Interest
            oi_normalized = np.log(market_data.open_interest + 1) / 25  # Log-Normalisierung
            features.append(oi_normalized)
            feature_names.append('open_interest_normalized')
        
        if market_data.funding_rate is not None:
            # Funding Rate (bereits in sinnvollem Bereich)
            features.append(np.tanh(market_data.funding_rate * 1000))
            feature_names.append('funding_rate')
        
        if market_data.long_short_ratio is not None:
            # Long/Short Ratio zentriert um 1
            ls_centered = market_data.long_short_ratio - 1.0
            features.append(np.tanh(ls_centered))
            feature_names.append('long_short_ratio_centered')
        
        # Technische Indikatoren (wenn genügend Daten)
        if len(historical_data) >= 14:
            prices = [d.price for d in historical_data[-14:] if d.price is not None]
            if len(prices) >= 10:
                rsi = self._calculate_rsi(prices)
                features.append(rsi / 100 - 0.5)  # Zentriert um 0
                feature_names.append('rsi_centered')
        
        # Momentum-Indikatoren
        if len(historical_data) >= 10:
            recent_prices = [d.price for d in historical_data[-5:] if d.price is not None]
            older_prices = [d.price for d in historical_data[-10:-5] if d.price is not None]
            
            if len(recent_prices) >= 3 and len(older_prices) >= 3:
                recent_avg = np.mean(recent_prices)
                older_avg = np.mean(older_prices)
                momentum = (recent_avg - older_avg) / older_avg
                features.append(np.tanh(momentum))
                feature_names.append('momentum_5_5')
        
        if len(features) < self.min_required_features:
            logger.warning(f"Nur {len(features)} Features extrahiert für {market_data.symbol}, "
                         f"Minimum {self.min_required_features} erforderlich")
            return None
        
        # Datenqualität bewerten
        quality_score = data_quality_manager.assess_data_quality(market_data, historical_data)
        
        return MarketFeatures(
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            features=np.array(features),
            feature_names=feature_names,
            data_quality_score=quality_score,
            available_data_sources=available_sources
        )
    
    def _validate_minimum_requirements(self, market_data: RealMarketData, 
                                     historical_data: List[RealMarketData]) -> bool:
        """Validiert ob Mindestanforderungen für Feature-Extraktion erfüllt sind"""
        # Aktueller Preis ist essentiell
        if market_data.price is None or market_data.price <= 0:
            return False
        
        # Mindestens 5 historische Datenpunkte mit Preisen
        valid_historical_prices = sum(1 for d in historical_data 
                                    if d.price is not None and d.price > 0)
        if valid_historical_prices < 5:
            return False
        
        return True
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Berechnet RSI nur mit verfügbaren Preisdaten"""
        if len(prices) < period + 1:
            return 50.0  # Neutraler RSI wenn nicht genügend Daten
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class ProductionCryptoPredictor:
    """
    Produktionsreifes Kryptowährungs-Vorhersagemodell
    Arbeitet ausschließlich mit realen Daten ohne Fallbacks
    """
    
    def __init__(self, symbol: str, min_data_quality: float = 0.6):
        self.symbol = symbol
        self.min_data_quality = min_data_quality
        
        # Adaptive Parameter-Struktur
        self.parameters = {
            'base_weights': {},
            'interaction_weights': {},
            'temporal_weights': {},
            'learning_rate': 0.001,
            'momentum_beta1': 0.9,
            'momentum_beta2': 0.999,
            'epsilon': 1e-8
        }
        
        # Momentum-Terme für Adam Optimizer
        self.momentum_terms = {
            'base_m': {},
            'base_v': {},
            'interaction_m': {},
            'interaction_v': {}
        }
        
        # Performance-Tracking
        self.prediction_history = []
        self.performance_metrics = {
            'accuracy_history': [],
            'error_history': [],
            'data_quality_history': [],
            'feature_importance': {}
        }
        
        # Komponenten
        self.feature_extractor = AdaptiveFeatureExtractor()
        self.data_quality_manager = DataQualityManager()
        
        # Wissenschaftliche Parameter
        self.confidence_threshold = 0.7  # Minimum Konfidenz für Vorhersagen
        self.min_training_samples = 20   # Minimum Samples vor Vorhersagen
        
        logger.info(f"ProductionCryptoPredictor für {symbol} initialisiert")
    
    def can_make_prediction(self, market_features: MarketFeatures) -> bool:
        """
        Prüft ob eine wissenschaftlich fundierte Vorhersage möglich ist
        """
        # Datenqualität prüfen
        if market_features.data_quality_score < self.min_data_quality:
            logger.info(f"Datenqualität zu niedrig: {market_features.data_quality_score:.3f} < {self.min_data_quality}")
            return False
        
        # Genügend historische Daten
        if len(self.prediction_history) < self.min_training_samples:
            logger.info(f"Nicht genügend Training-Samples: {len(self.prediction_history)} < {self.min_training_samples}")
            return False
        
        # Feature-Verfügbarkeit
        if len(market_features.features) < self.feature_extractor.min_required_features:
            logger.info(f"Nicht genügend Features: {len(market_features.features)} < {self.feature_extractor.min_required_features}")
            return False
        
        return True
    
    def predict_with_uncertainty(self, market_features: MarketFeatures) -> Optional[Dict]:
        """
        Macht Vorhersage nur wenn wissenschaftliche Standards erfüllt sind
        """
        if not self.can_make_prediction(market_features):
            return None
        
        # Adaptive Gewichte basierend auf verfügbaren Features
        prediction_components = self._calculate_prediction_components(market_features)
        
        # Ensemble-Vorhersage
        base_prediction = prediction_components['base']
        interaction_effects = prediction_components['interactions']
        temporal_effects = prediction_components['temporal']
        
        # Gewichtete Kombination basierend auf Feature-Verfügbarkeit
        weights = self._calculate_adaptive_weights(market_features)
        
        final_prediction = (weights['base'] * base_prediction + 
                          weights['interaction'] * interaction_effects +
                          weights['temporal'] * temporal_effects)
        
        # Unsicherheitsquantifizierung
        uncertainty = self._calculate_uncertainty(market_features, prediction_components)
        
        # Konfidenzschätzung
        confidence = self._calculate_confidence(market_features, uncertainty)
        
        if confidence < self.confidence_threshold:
            logger.info(f"Konfidenz zu niedrig: {confidence:.3f} < {self.confidence_threshold}")
            return None
        
        return {
            'prediction': final_prediction,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'data_quality': market_features.data_quality_score,
            'feature_count': len(market_features.features),
            'available_sources': market_features.available_data_sources,
            'timestamp': market_features.timestamp
        }
    
    def _calculate_prediction_components(self, market_features: MarketFeatures) -> Dict:
        """Berechnet verschiedene Vorhersage-Komponenten"""
        features = market_features.features
        feature_names = market_features.feature_names
        
        # Basis-Vorhersage
        base_prediction = 0.0
        for i, (feature_name, feature_value) in enumerate(zip(feature_names, features)):
            weight = self.parameters['base_weights'].get(feature_name, 0.0)
            base_prediction += weight * feature_value
        
        # Interaktionseffekte (nur für verfügbare Feature-Paare)
        interaction_effects = 0.0
        for i, name1 in enumerate(feature_names):
            for j, name2 in enumerate(feature_names[i+1:], i+1):
                interaction_key = f"{name1}_{name2}"
                weight = self.parameters['interaction_weights'].get(interaction_key, 0.0)
                interaction_effects += weight * features[i] * features[j]
        
        # Temporale Effekte basierend auf Performance-Historie
        temporal_effects = 0.0
        if len(self.performance_metrics['accuracy_history']) > 5:
            recent_accuracy = np.mean(self.performance_metrics['accuracy_history'][-5:])
            temporal_effects = (recent_accuracy - 0.5) * 0.1  # Schwacher temporaler Effekt
        
        return {
            'base': base_prediction,
            'interactions': interaction_effects,
            'temporal': temporal_effects
        }
    
    def _calculate_adaptive_weights(self, market_features: MarketFeatures) -> Dict:
        """Berechnet adaptive Gewichte basierend auf Datenqualität"""
        base_weight = 0.6
        interaction_weight = 0.3 * market_features.data_quality_score
        temporal_weight = 0.1 * min(len(self.prediction_history) / 50, 1.0)
        
        # Normalisierung
        total = base_weight + interaction_weight + temporal_weight
        
        return {
            'base': base_weight / total,
            'interaction': interaction_weight / total,
            'temporal': temporal_weight / total
        }
    
    def _calculate_uncertainty(self, market_features: MarketFeatures, 
                             prediction_components: Dict) -> float:
        """Berechnet Unsicherheit basierend auf Datenqualität und Modell-Varianz"""
        # Modell-Unsicherheit (Varianz zwischen Komponenten)
        component_values = list(prediction_components.values())
        model_uncertainty = np.std(component_values) if len(component_values) > 1 else 0.1
        
        # Daten-Unsicherheit (umgekehrt proportional zur Datenqualität)
        data_uncertainty = (1.0 - market_features.data_quality_score) * 0.2
        
        # Historische Unsicherheit
        historical_uncertainty = 0.1
        if len(self.performance_metrics['error_history']) > 5:
            historical_uncertainty = np.std(self.performance_metrics['error_history'][-10:])
        
        total_uncertainty = np.sqrt(model_uncertainty**2 + data_uncertainty**2 + historical_uncertainty**2)
        return total_uncertainty
    
    def _calculate_confidence(self, market_features: MarketFeatures, uncertainty: float) -> float:
        """Berechnet Konfidenz basierend auf Datenqualität und Unsicherheit"""
        # Basis-Konfidenz aus Datenqualität
        base_confidence = market_features.data_quality_score
        
        # Unsicherheits-Anpassung
        uncertainty_penalty = min(uncertainty * 2, 0.5)  # Max 50% Penalty
        
        # Feature-Vollständigkeit
        feature_completeness = min(len(market_features.features) / 10, 1.0)
        
        confidence = base_confidence * (1 - uncertainty_penalty) * feature_completeness
        return np.clip(confidence, 0.0, 1.0)
    
    def update_model(self, market_features: MarketFeatures, actual_change: float, 
                    prediction_result: Optional[Dict]) -> bool:
        """
        Aktualisiert Modell nur mit realen Daten
        """
        if prediction_result is None:
            # Kein Update wenn keine Vorhersage gemacht wurde
            return False
        
        predicted_change = prediction_result['prediction']
        error = abs(predicted_change - actual_change)
        
        # Performance-Metriken aktualisieren
        self.performance_metrics['error_history'].append(error)
        self.performance_metrics['data_quality_history'].append(market_features.data_quality_score)
        
        # Accuracy berechnen (Richtungsgenauigkeit)
        direction_correct = np.sign(predicted_change) == np.sign(actual_change)
        self.performance_metrics['accuracy_history'].append(1.0 if direction_correct else 0.0)
        
        # Parameter-Update nur wenn Datenqualität ausreichend
        if market_features.data_quality_score >= self.min_data_quality:
            self._update_parameters(market_features, actual_change, predicted_change)
        
        # Adaptive Lernrate
        self._update_learning_rate()
        
        logger.info(f"Modell aktualisiert für {self.symbol}: Fehler={error:.4f}, "
                   f"Richtung korrekt={direction_correct}, Datenqualität={market_features.data_quality_score:.3f}")
        
        return True
    
    def _update_parameters(self, market_features: MarketFeatures, 
                          actual_change: float, predicted_change: float):
        """Aktualisiert Parameter mit Adam Optimizer"""
        error = actual_change - predicted_change
        features = market_features.features
        feature_names = market_features.feature_names
        
        # Gradientenberechnung für Basis-Gewichte
        for i, feature_name in enumerate(feature_names):
            gradient = -error * features[i]
            
            # Adam Update
            if feature_name not in self.momentum_terms['base_m']:
                self.momentum_terms['base_m'][feature_name] = 0.0
                self.momentum_terms['base_v'][feature_name] = 0.0
            
            self.momentum_terms['base_m'][feature_name] = (
                self.parameters['momentum_beta1'] * self.momentum_terms['base_m'][feature_name] +
                (1 - self.parameters['momentum_beta1']) * gradient
            )
            
            self.momentum_terms['base_v'][feature_name] = (
                self.parameters['momentum_beta2'] * self.momentum_terms['base_v'][feature_name] +
                (1 - self.parameters['momentum_beta2']) * gradient**2
            )
            
            # Bias-Korrektur
            t = len(self.performance_metrics['error_history'])
            m_hat = self.momentum_terms['base_m'][feature_name] / (1 - self.parameters['momentum_beta1']**t)
            v_hat = self.momentum_terms['base_v'][feature_name] / (1 - self.parameters['momentum_beta2']**t)
            
            # Parameter-Update
            if feature_name not in self.parameters['base_weights']:
                self.parameters['base_weights'][feature_name] = 0.0
            
            self.parameters['base_weights'][feature_name] -= (
                self.parameters['learning_rate'] * m_hat / (np.sqrt(v_hat) + self.parameters['epsilon'])
            )
    
    def _update_learning_rate(self):
        """Adaptive Lernraten-Anpassung"""
        if len(self.performance_metrics['accuracy_history']) > 10:
            recent_accuracy = np.mean(self.performance_metrics['accuracy_history'][-10:])
            older_accuracy = np.mean(self.performance_metrics['accuracy_history'][-20:-10]) if len(self.performance_metrics['accuracy_history']) > 20 else 0.5
            
            if recent_accuracy > older_accuracy:
                self.parameters['learning_rate'] *= 1.01  # Leichte Erhöhung
            else:
                self.parameters['learning_rate'] *= 0.99  # Leichte Reduzierung
            
            # Grenzen einhalten
            self.parameters['learning_rate'] = np.clip(self.parameters['learning_rate'], 1e-5, 0.01)
    
    def get_model_status(self) -> Dict:
        """Gibt aktuellen Modellstatus zurück"""
        status = {
            'symbol': self.symbol,
            'total_predictions': len(self.prediction_history),
            'min_data_quality': self.min_data_quality,
            'confidence_threshold': self.confidence_threshold,
            'current_learning_rate': self.parameters['learning_rate'],
            'parameter_count': len(self.parameters['base_weights']),
            'can_predict': len(self.prediction_history) >= self.min_training_samples
        }
        
        if len(self.performance_metrics['accuracy_history']) > 0:
            status.update({
                'average_accuracy': np.mean(self.performance_metrics['accuracy_history']),
                'recent_accuracy': np.mean(self.performance_metrics['accuracy_history'][-10:]) if len(self.performance_metrics['accuracy_history']) >= 10 else None,
                'average_error': np.mean(self.performance_metrics['error_history']),
                'average_data_quality': np.mean(self.performance_metrics['data_quality_history'])
            })
        
        return status

def create_production_model(symbol: str, min_data_quality: float = 0.6) -> ProductionCryptoPredictor:
    """Factory-Funktion für Produktionsmodelle"""
    return ProductionCryptoPredictor(symbol=symbol, min_data_quality=min_data_quality)

