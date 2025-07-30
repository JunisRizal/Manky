#!/usr/bin/env python3
"""
Flexibles Produktionsmodell mit reduzierten Datenanforderungen
Wissenschaftlich robuste Implementierung für reale Marktbedingungen
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

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

class FlexibleDataQualityManager:
    """Flexibler Datenqualitäts-Manager mit reduzierten Anforderungen"""
    
    def __init__(self):
        self.essential_features = ['price']  # Nur Preis ist wirklich essentiell
        self.beneficial_features = ['volume', 'price_change', 'volatility']
        
    def assess_data_quality(self, market_data: RealMarketData, 
                          historical_data: List[RealMarketData]) -> float:
        """Bewertet Datenqualität mit flexiblen Standards"""
        quality_score = 0.0
        
        # Basis-Datenqualität (Preis ist essentiell)
        if market_data.price is not None and market_data.price > 0:
            quality_score += 0.5  # 50% für gültigen Preis
        else:
            return 0.0  # Ohne Preis keine Vorhersage möglich
        
        # Historische Daten-Kontinuität (flexibler)
        if len(historical_data) >= 5:  # Reduziert von 10
            recent_data_points = sum(1 for d in historical_data[-5:] 
                                   if d.price is not None and d.price > 0)
            continuity_score = recent_data_points / 5
            quality_score += 0.3 * continuity_score
        elif len(historical_data) >= 2:
            # Auch mit wenigen Daten arbeiten
            recent_data_points = sum(1 for d in historical_data 
                                   if d.price is not None and d.price > 0)
            continuity_score = recent_data_points / len(historical_data)
            quality_score += 0.2 * continuity_score
        
        # Volumen-Verfügbarkeit (optional)
        if market_data.volume is not None and market_data.volume > 0:
            quality_score += 0.1
        
        # Erweiterte Daten (optional)
        extended_data_count = sum([
            1 if market_data.open_interest is not None else 0,
            1 if market_data.funding_rate is not None else 0,
            1 if market_data.long_short_ratio is not None else 0
        ])
        quality_score += 0.1 * (extended_data_count / 3)
        
        return min(quality_score, 1.0)
    
    def get_available_features(self, market_data: RealMarketData) -> List[str]:
        """Ermittelt verfügbare Features"""
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

class FlexibleFeatureExtractor:
    """Flexibler Feature-Extractor mit minimalen Anforderungen"""
    
    def __init__(self):
        self.min_required_features = 2  # Reduziert von 3
        self.feature_registry = {}
        
    def extract_features(self, market_data: RealMarketData, 
                        historical_data: List[RealMarketData],
                        data_quality_manager: FlexibleDataQualityManager) -> Optional[MarketFeatures]:
        """
        Extrahiert Features mit minimalen Anforderungen
        """
        if not self._validate_minimum_requirements(market_data, historical_data):
            return None
        
        features = []
        feature_names = []
        available_sources = data_quality_manager.get_available_features(market_data)
        
        # Basis-Features (immer verfügbar wenn Preis vorhanden)
        if market_data.price is not None:
            # Preisbasierte Features
            if len(historical_data) > 0 and historical_data[-1].price is not None:
                price_change = (market_data.price - historical_data[-1].price) / historical_data[-1].price
                features.append(np.tanh(price_change * 10))  # Normalisiert
                feature_names.append('price_change_1period')
            
            # Preis-Momentum (wenn mindestens 3 historische Punkte)
            if len(historical_data) >= 3:
                recent_prices = [d.price for d in historical_data[-3:] if d.price is not None]
                if len(recent_prices) >= 2:
                    momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    features.append(np.tanh(momentum * 5))
                    feature_names.append('momentum_3period')
            
            # Volatilität (wenn mindestens 5 historische Punkte)
            if len(historical_data) >= 5:
                prices = [d.price for d in historical_data[-5:] if d.price is not None]
                if len(prices) >= 3:
                    volatility = np.std(prices) / np.mean(prices)
                    features.append(min(volatility * 10, 1.0))  # Begrenzt auf 1
                    feature_names.append('volatility_5period')
        
        # Volumen-Features (optional)
        if market_data.volume is not None and len(historical_data) > 0:
            last_volume = None
            for d in reversed(historical_data):
                if d.volume is not None:
                    last_volume = d.volume
                    break
            
            if last_volume is not None and last_volume > 0:
                volume_change = (market_data.volume - last_volume) / last_volume
                features.append(np.tanh(volume_change))
                feature_names.append('volume_change')
        
        # Erweiterte Features (falls verfügbar)
        if market_data.funding_rate is not None:
            features.append(np.tanh(market_data.funding_rate * 1000))
            feature_names.append('funding_rate')
        
        if market_data.long_short_ratio is not None:
            ls_centered = market_data.long_short_ratio - 1.0
            features.append(np.tanh(ls_centered))
            feature_names.append('long_short_ratio_centered')
        
        # Zeitbasierte Features
        hour_of_day = datetime.fromtimestamp(market_data.timestamp).hour
        features.append(np.sin(2 * np.pi * hour_of_day / 24))  # Zyklisch
        feature_names.append('hour_sin')
        
        day_of_week = datetime.fromtimestamp(market_data.timestamp).weekday()
        features.append(np.sin(2 * np.pi * day_of_week / 7))  # Zyklisch
        feature_names.append('day_sin')
        
        # Prüfe Mindestanforderungen
        if len(features) < self.min_required_features:
            logger.warning(f"Nur {len(features)} Features für {market_data.symbol}, "
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
        """Validiert Mindestanforderungen (sehr flexibel)"""
        # Aktueller Preis ist essentiell
        if market_data.price is None or market_data.price <= 0:
            return False
        
        # Mindestens 2 historische Datenpunkte mit Preisen (reduziert von 5)
        valid_historical_prices = sum(1 for d in historical_data 
                                    if d.price is not None and d.price > 0)
        if valid_historical_prices < 2:
            return False
        
        return True

class FlexibleCryptoPredictor:
    """
    Flexibles Kryptowährungs-Vorhersagemodell mit reduzierten Anforderungen
    """
    
    def __init__(self, symbol: str, min_data_quality: float = 0.4):
        self.symbol = symbol
        self.min_data_quality = min_data_quality  # Reduziert von 0.6
        
        # Adaptive Parameter-Struktur
        self.parameters = {
            'base_weights': {},
            'interaction_weights': {},
            'learning_rate': 0.01,  # Höhere Lernrate für schnellere Anpassung
            'momentum_beta1': 0.9,
            'momentum_beta2': 0.999,
            'epsilon': 1e-8
        }
        
        # Momentum-Terme
        self.momentum_terms = {
            'base_m': {},
            'base_v': {}
        }
        
        # Performance-Tracking
        self.prediction_history = []
        self.performance_metrics = {
            'accuracy_history': [],
            'error_history': [],
            'data_quality_history': [],
            'confidence_history': []
        }
        
        # Komponenten
        self.feature_extractor = FlexibleFeatureExtractor()
        self.data_quality_manager = FlexibleDataQualityManager()
        
        # Wissenschaftliche Parameter (flexibler)
        self.confidence_threshold = 0.5  # Reduziert von 0.7
        self.min_training_samples = 5    # Reduziert von 20
        
        logger.info(f"FlexibleCryptoPredictor für {symbol} initialisiert")
    
    def can_make_prediction(self, market_features: MarketFeatures) -> bool:
        """Prüft Vorhersagemöglichkeit mit flexiblen Standards"""
        # Datenqualität prüfen
        if market_features.data_quality_score < self.min_data_quality:
            logger.debug(f"Datenqualität zu niedrig: {market_features.data_quality_score:.3f} < {self.min_data_quality}")
            return False
        
        # Genügend historische Daten (flexibler)
        if len(self.prediction_history) < self.min_training_samples:
            logger.debug(f"Nicht genügend Training-Samples: {len(self.prediction_history)} < {self.min_training_samples}")
            return False
        
        # Feature-Verfügbarkeit
        if len(market_features.features) < self.feature_extractor.min_required_features:
            logger.debug(f"Nicht genügend Features: {len(market_features.features)} < {self.feature_extractor.min_required_features}")
            return False
        
        return True
    
    def predict_with_uncertainty(self, market_features: MarketFeatures) -> Optional[Dict]:
        """Macht Vorhersage mit flexiblen Standards"""
        if not self.can_make_prediction(market_features):
            return None
        
        features = market_features.features
        feature_names = market_features.feature_names
        
        # Basis-Vorhersage
        base_prediction = 0.0
        for i, (feature_name, feature_value) in enumerate(zip(feature_names, features)):
            weight = self.parameters['base_weights'].get(feature_name, 0.0)
            base_prediction += weight * feature_value
        
        # Einfache Interaktionseffekte (nur für erste 3 Features)
        interaction_effects = 0.0
        if len(features) >= 2:
            for i in range(min(3, len(features))):
                for j in range(i+1, min(3, len(features))):
                    interaction_key = f"{feature_names[i]}_{feature_names[j]}"
                    weight = self.parameters['interaction_weights'].get(interaction_key, 0.0)
                    interaction_effects += weight * features[i] * features[j]
        
        # Ensemble-Vorhersage (einfacher)
        final_prediction = 0.7 * base_prediction + 0.3 * interaction_effects
        
        # Unsicherheitsquantifizierung
        uncertainty = self._calculate_uncertainty(market_features)
        
        # Konfidenzschätzung
        confidence = self._calculate_confidence(market_features, uncertainty)
        
        if confidence < self.confidence_threshold:
            logger.debug(f"Konfidenz zu niedrig: {confidence:.3f} < {self.confidence_threshold}")
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
    
    def _calculate_uncertainty(self, market_features: MarketFeatures) -> float:
        """Berechnet Unsicherheit"""
        # Basis-Unsicherheit basierend auf Datenqualität
        data_uncertainty = (1.0 - market_features.data_quality_score) * 0.3
        
        # Modell-Unsicherheit basierend auf Erfahrung
        model_uncertainty = 0.2
        if len(self.performance_metrics['error_history']) > 3:
            recent_errors = self.performance_metrics['error_history'][-5:]
            model_uncertainty = np.std(recent_errors) if len(recent_errors) > 1 else 0.2
        
        # Feature-Unsicherheit
        feature_uncertainty = max(0.1, 0.3 - len(market_features.features) * 0.05)
        
        total_uncertainty = np.sqrt(data_uncertainty**2 + model_uncertainty**2 + feature_uncertainty**2)
        return min(total_uncertainty, 0.5)  # Maximal 50% Unsicherheit
    
    def _calculate_confidence(self, market_features: MarketFeatures, uncertainty: float) -> float:
        """Berechnet Konfidenz"""
        # Basis-Konfidenz aus Datenqualität
        base_confidence = market_features.data_quality_score
        
        # Erfahrungs-Bonus
        experience_bonus = min(len(self.prediction_history) / 20, 0.2)
        
        # Unsicherheits-Penalty
        uncertainty_penalty = uncertainty * 0.5
        
        confidence = base_confidence + experience_bonus - uncertainty_penalty
        return np.clip(confidence, 0.0, 1.0)
    
    def update_model(self, market_features: MarketFeatures, actual_change: float, 
                    prediction_result: Optional[Dict]) -> bool:
        """Aktualisiert Modell mit realen Daten"""
        if prediction_result is None:
            # Auch ohne Vorhersage können wir lernen (für zukünftige Vorhersagen)
            self._update_without_prediction(market_features, actual_change)
            return True
        
        predicted_change = prediction_result['prediction']
        error = abs(predicted_change - actual_change)
        
        # Performance-Metriken aktualisieren
        self.performance_metrics['error_history'].append(error)
        self.performance_metrics['data_quality_history'].append(market_features.data_quality_score)
        self.performance_metrics['confidence_history'].append(prediction_result['confidence'])
        
        # Accuracy berechnen
        direction_correct = np.sign(predicted_change) == np.sign(actual_change)
        self.performance_metrics['accuracy_history'].append(1.0 if direction_correct else 0.0)
        
        # Parameter-Update
        self._update_parameters(market_features, actual_change, predicted_change)
        
        # Adaptive Lernrate
        self._update_learning_rate()
        
        # Prediction History aktualisieren
        self.prediction_history.append({
            'timestamp': market_features.timestamp,
            'predicted': predicted_change,
            'actual': actual_change,
            'error': error,
            'confidence': prediction_result['confidence']
        })
        
        logger.info(f"Modell für {self.symbol} aktualisiert: Fehler={error:.4f}, "
                   f"Richtung korrekt={direction_correct}")
        
        return True
    
    def _update_without_prediction(self, market_features: MarketFeatures, actual_change: float):
        """Lernt auch ohne Vorhersage für zukünftige Verbesserungen"""
        # Einfaches Update der Basis-Gewichte
        features = market_features.features
        feature_names = market_features.feature_names
        
        # Schwaches Signal für zukünftige Vorhersagen
        for i, feature_name in enumerate(feature_names):
            if feature_name not in self.parameters['base_weights']:
                self.parameters['base_weights'][feature_name] = 0.0
            
            # Sehr kleine Anpassung basierend auf Korrelation
            correlation = features[i] * actual_change
            self.parameters['base_weights'][feature_name] += 0.001 * correlation
        
        logger.debug(f"Passives Lernen für {self.symbol}: {len(feature_names)} Features")
    
    def _update_parameters(self, market_features: MarketFeatures, 
                          actual_change: float, predicted_change: float):
        """Aktualisiert Parameter mit Adam Optimizer"""
        error = actual_change - predicted_change
        features = market_features.features
        feature_names = market_features.feature_names
        
        # Gradientenberechnung für Basis-Gewichte
        for i, feature_name in enumerate(feature_names):
            gradient = -error * features[i]
            
            # Initialisiere Momentum-Terme falls nötig
            if feature_name not in self.momentum_terms['base_m']:
                self.momentum_terms['base_m'][feature_name] = 0.0
                self.momentum_terms['base_v'][feature_name] = 0.0
            
            # Adam Update
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
        if len(self.performance_metrics['accuracy_history']) > 5:
            recent_accuracy = np.mean(self.performance_metrics['accuracy_history'][-5:])
            
            if recent_accuracy > 0.6:
                self.parameters['learning_rate'] *= 1.02  # Leichte Erhöhung
            elif recent_accuracy < 0.4:
                self.parameters['learning_rate'] *= 0.98  # Leichte Reduzierung
            
            # Grenzen einhalten
            self.parameters['learning_rate'] = np.clip(self.parameters['learning_rate'], 1e-4, 0.05)
    
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
                'recent_accuracy': np.mean(self.performance_metrics['accuracy_history'][-5:]) if len(self.performance_metrics['accuracy_history']) >= 5 else None,
                'average_error': np.mean(self.performance_metrics['error_history']),
                'average_confidence': np.mean(self.performance_metrics['confidence_history']),
                'average_data_quality': np.mean(self.performance_metrics['data_quality_history'])
            })
        
        return status

def create_flexible_model(symbol: str, min_data_quality: float = 0.4) -> FlexibleCryptoPredictor:
    """Factory-Funktion für flexible Produktionsmodelle"""
    return FlexibleCryptoPredictor(symbol=symbol, min_data_quality=min_data_quality)

