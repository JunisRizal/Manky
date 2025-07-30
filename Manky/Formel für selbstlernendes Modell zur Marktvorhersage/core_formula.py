#!/usr/bin/env python3
"""
Kernformel für das Selbstlernende Kryptowährungs-Vorhersagemodell
Erweiterte Version des TrustLogIQ Systems mit modernen ML-Ansätzen
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.special import softmax
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketData:
    """Struktur für Marktdaten"""
    timestamp: float
    price: float
    volume: float
    open_interest: float
    funding_rate: float
    long_short_ratio: float
    fear_greed_index: float
    sentiment_score: float
    liquidations: float
    whale_alerts: int
    
class AdaptiveCryptoPredictor:
    """
    Selbstlernendes Kryptowährungs-Vorhersagemodell
    Basiert auf erweiterten Korrelationsmatrizen und Bayesianischen Ansätzen
    """
    
    def __init__(self, n_features: int = 25, n_regimes: int = 3):
        self.n_features = n_features
        self.n_regimes = n_regimes
        
        # Parameter-Matrizen initialisieren
        self.theta_base = np.random.normal(0, 0.1, (n_features,))
        self.theta_interaction = np.random.normal(0, 0.05, (n_features, n_features))
        self.theta_temporal = np.random.normal(0, 0.1, (n_features, 10))  # 10 Zeitschritte
        self.theta_regime = np.random.normal(0, 0.1, (n_regimes, n_features))
        
        # Adaptive Parameter
        self.learning_rate = 0.001
        self.momentum_beta1 = 0.9
        self.momentum_beta2 = 0.999
        self.epsilon = 1e-8
        
        # Momentum-Terme für Adam Optimizer
        self.m_base = np.zeros_like(self.theta_base)
        self.v_base = np.zeros_like(self.theta_base)
        self.m_interaction = np.zeros_like(self.theta_interaction)
        self.v_interaction = np.zeros_like(self.theta_interaction)
        
        # Unsicherheitsparameter
        self.uncertainty_alpha = 1.0
        self.uncertainty_beta = 1.0
        
        # Performance-Tracking
        self.prediction_history = []
        self.performance_scores = []
        self.regime_probabilities = np.ones(n_regimes) / n_regimes
        
    def extract_features(self, market_data: MarketData, 
                        historical_data: List[MarketData]) -> np.ndarray:
        """
        Extrahiert erweiterte Features aus Marktdaten
        """
        features = []
        
        # Basis-Features (normalisiert)
        features.extend([
            market_data.open_interest / 1e11,  # β1
            market_data.funding_rate * 1000,   # β2
            abs(market_data.price - historical_data[-1].price) / historical_data[-1].price if historical_data else 0,  # β3
            market_data.long_short_ratio - 1.0,  # β4
            market_data.volume / 1e10,  # β5
            market_data.fear_greed_index / 100,  # β6
            market_data.sentiment_score,  # β7
            market_data.liquidations / 1e9,  # β8
            market_data.whale_alerts / 10,  # β9
        ])
        
        # Technische Indikatoren (simuliert)
        if len(historical_data) >= 14:
            prices = [d.price for d in historical_data[-14:]]
            rsi = self._calculate_rsi(prices)
            features.append(rsi / 100)  # β10
        else:
            features.append(0.5)
            
        # Volatilität
        if len(historical_data) >= 20:
            prices = [d.price for d in historical_data[-20:]]
            volatility = np.std(prices) / np.mean(prices)
            features.append(volatility)  # β11
        else:
            features.append(0.1)
            
        # Momentum-Indikatoren
        if len(historical_data) >= 5:
            short_ma = np.mean([d.price for d in historical_data[-5:]])
            long_ma = np.mean([d.price for d in historical_data[-20:]]) if len(historical_data) >= 20 else short_ma
            momentum = (short_ma - long_ma) / long_ma if long_ma > 0 else 0
            features.append(momentum)  # β12
        else:
            features.append(0)
            
        # Volume-Profile
        if len(historical_data) >= 10:
            volumes = [d.volume for d in historical_data[-10:]]
            volume_trend = (volumes[-1] - np.mean(volumes)) / np.mean(volumes) if np.mean(volumes) > 0 else 0
            features.append(volume_trend)  # β13
        else:
            features.append(0)
            
        # Erweiterte Features (bis zu 25 Features)
        while len(features) < self.n_features:
            # Interaktionseffekte und abgeleitete Metriken
            if len(features) >= 2:
                features.append(features[0] * features[1])  # Interaktion
            else:
                features.append(0)
                
        return np.array(features[:self.n_features])
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Berechnet den Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
            
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
    
    def predict_with_uncertainty(self, features: np.ndarray, 
                                historical_features: List[np.ndarray]) -> Tuple[float, float, float]:
        """
        Hauptvorhersagefunktion mit Unsicherheitsquantifizierung
        
        Returns:
            prediction: Vorhersagewert
            epistemic_uncertainty: Modell-Unsicherheit
            aleatoric_uncertainty: Daten-Unsicherheit
        """
        
        # 1. Basis-Vorhersage
        base_prediction = np.dot(self.theta_base, features)
        
        # 2. Interaktionseffekte
        interaction_effects = 0
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                interaction_effects += self.theta_interaction[i,j] * features[i] * features[j]
        
        # 3. Temporale Effekte (LSTM-ähnlich)
        temporal_effects = 0
        if len(historical_features) >= 10:
            recent_features = np.array(historical_features[-10:])
            for t in range(len(recent_features)):
                decay_factor = 0.9 ** (len(recent_features) - t - 1)
                temporal_effects += decay_factor * np.dot(self.theta_temporal[:, t], recent_features[t])
        
        # 4. Regime-spezifische Vorhersage
        regime_predictions = []
        for r in range(self.n_regimes):
            regime_pred = np.dot(self.theta_regime[r], features)
            regime_predictions.append(regime_pred)
        
        # Gewichtete Regime-Vorhersage
        regime_prediction = np.dot(self.regime_probabilities, regime_predictions)
        
        # 5. Ensemble-Vorhersage
        ensemble_weights = softmax([0.4, 0.2, 0.2, 0.2])  # Gewichte für verschiedene Komponenten
        final_prediction = (ensemble_weights[0] * base_prediction + 
                          ensemble_weights[1] * interaction_effects +
                          ensemble_weights[2] * temporal_effects +
                          ensemble_weights[3] * regime_prediction)
        
        # 6. Unsicherheitsquantifizierung
        
        # Epistemische Unsicherheit (Modell-Unsicherheit)
        model_variance = np.var([base_prediction, interaction_effects, 
                               temporal_effects, regime_prediction])
        epistemic_uncertainty = np.sqrt(model_variance)
        
        # Aleatorische Unsicherheit (Daten-Unsicherheit)
        # Basiert auf historischer Vorhersagegenauigkeit
        if len(self.performance_scores) > 0:
            recent_errors = self.performance_scores[-20:] if len(self.performance_scores) >= 20 else self.performance_scores
            aleatoric_uncertainty = np.std(recent_errors) if len(recent_errors) > 1 else 0.1
        else:
            aleatoric_uncertainty = 0.1
            
        return final_prediction, epistemic_uncertainty, aleatoric_uncertainty
    
    def update_parameters(self, features: np.ndarray, actual_value: float, 
                         predicted_value: float, learning_rate: Optional[float] = None):
        """
        Adaptive Parameteraktualisierung mit Adam Optimizer
        """
        if learning_rate is None:
            learning_rate = self.learning_rate
            
        # Berechne Prediction Error
        error = actual_value - predicted_value
        self.performance_scores.append(abs(error))
        
        # Gradient für Basis-Parameter
        grad_base = -error * features
        
        # Adam Update für Basis-Parameter
        self.m_base = self.momentum_beta1 * self.m_base + (1 - self.momentum_beta1) * grad_base
        self.v_base = self.momentum_beta2 * self.v_base + (1 - self.momentum_beta2) * (grad_base ** 2)
        
        # Bias-Korrektur
        m_hat = self.m_base / (1 - self.momentum_beta1 ** (len(self.performance_scores)))
        v_hat = self.v_base / (1 - self.momentum_beta2 ** (len(self.performance_scores)))
        
        # Parameter-Update
        self.theta_base -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Regime-Wahrscheinlichkeiten aktualisieren
        self._update_regime_probabilities(error)
        
        # Adaptive Learning Rate
        if len(self.performance_scores) > 10:
            recent_performance = np.mean(self.performance_scores[-10:])
            if recent_performance > np.mean(self.performance_scores[:-10]) if len(self.performance_scores) > 20 else 0.1:
                self.learning_rate *= 0.95  # Reduziere Learning Rate bei schlechter Performance
            else:
                self.learning_rate *= 1.01  # Erhöhe Learning Rate bei guter Performance
                
        # Clipping für Stabilität
        self.learning_rate = np.clip(self.learning_rate, 1e-5, 0.01)
    
    def _update_regime_probabilities(self, error: float):
        """Aktualisiert Regime-Wahrscheinlichkeiten basierend auf Prediction Error"""
        # Einfache Heuristik: Hoher Fehler deutet auf Regime-Wechsel hin
        if abs(error) > np.std(self.performance_scores[-20:]) * 2 if len(self.performance_scores) >= 20 else 0.1:
            # Regime-Wechsel erkannt - gleichmäßige Verteilung
            self.regime_probabilities = np.ones(self.n_regimes) / self.n_regimes
        else:
            # Aktuelles Regime verstärken
            best_regime = np.argmax(self.regime_probabilities)
            self.regime_probabilities *= 0.95
            self.regime_probabilities[best_regime] += 0.05
            
        # Normalisierung
        self.regime_probabilities /= np.sum(self.regime_probabilities)
    
    def calculate_confidence_interval(self, prediction: float, 
                                    epistemic_uncertainty: float,
                                    aleatoric_uncertainty: float,
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Berechnet Konfidenzintervall für Vorhersage
        """
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        lower_bound = prediction - z_score * total_uncertainty
        upper_bound = prediction + z_score * total_uncertainty
        
        return lower_bound, upper_bound
    
    def get_model_summary(self) -> Dict:
        """Gibt eine Zusammenfassung des Modellzustands zurück"""
        return {
            'n_predictions': len(self.performance_scores),
            'avg_error': np.mean(self.performance_scores) if self.performance_scores else 0,
            'current_learning_rate': self.learning_rate,
            'regime_probabilities': self.regime_probabilities.tolist(),
            'model_stability': np.std(self.performance_scores[-10:]) if len(self.performance_scores) >= 10 else 0
        }

def demonstrate_model():
    """Demonstriert die Verwendung des Modells"""
    print("=== Selbstlernendes Kryptowährungs-Vorhersagemodell ===")
    print()
    
    # Modell initialisieren
    model = AdaptiveCryptoPredictor(n_features=25, n_regimes=3)
    
    # Simulierte Marktdaten generieren
    np.random.seed(42)
    historical_data = []
    historical_features = []
    
    base_price = 50000  # BTC Startpreis
    
    print("Simuliere Marktdaten und Vorhersagen...")
    print()
    
    for i in range(100):
        # Simulierte Marktdaten
        price_change = np.random.normal(0, 0.02)  # 2% Volatilität
        new_price = base_price * (1 + price_change)
        
        market_data = MarketData(
            timestamp=i,
            price=new_price,
            volume=np.random.uniform(1e9, 5e9),
            open_interest=np.random.uniform(1e10, 3e11),
            funding_rate=np.random.normal(0.0001, 0.0005),
            long_short_ratio=np.random.uniform(0.8, 1.2),
            fear_greed_index=np.random.uniform(20, 80),
            sentiment_score=np.random.normal(0, 0.3),
            liquidations=np.random.uniform(1e8, 1e10),
            whale_alerts=np.random.poisson(5)
        )
        
        # Features extrahieren
        features = model.extract_features(market_data, historical_data)
        
        # Vorhersage machen (ab dem 10. Datenpunkt)
        if i >= 10:
            prediction, epistemic_unc, aleatoric_unc = model.predict_with_uncertainty(
                features, historical_features[-10:]
            )
            
            # Konfidenzintervall berechnen
            lower, upper = model.calculate_confidence_interval(
                prediction, epistemic_unc, aleatoric_unc
            )
            
            # Nächsten "tatsächlichen" Wert simulieren
            actual_next_change = np.random.normal(0, 0.02)
            actual_next_price = new_price * (1 + actual_next_change)
            actual_change = (actual_next_price - new_price) / new_price
            
            # Modell mit tatsächlichem Wert aktualisieren
            model.update_parameters(features, actual_change, prediction)
            
            # Ergebnisse alle 20 Schritte ausgeben
            if i % 20 == 0:
                print(f"Schritt {i}:")
                print(f"  Aktueller Preis: ${new_price:,.2f}")
                print(f"  Vorhersage Änderung: {prediction:.4f} ({prediction*100:.2f}%)")
                print(f"  Tatsächliche Änderung: {actual_change:.4f} ({actual_change*100:.2f}%)")
                print(f"  Konfidenzintervall: [{lower:.4f}, {upper:.4f}]")
                print(f"  Epistemische Unsicherheit: {epistemic_unc:.4f}")
                print(f"  Aleatorische Unsicherheit: {aleatoric_unc:.4f}")
                print(f"  Vorhersagefehler: {abs(prediction - actual_change):.4f}")
                print()
        
        # Daten für nächste Iteration speichern
        historical_data.append(market_data)
        historical_features.append(features)
        base_price = new_price
    
    # Modell-Zusammenfassung
    summary = model.get_model_summary()
    print("=== Modell-Zusammenfassung ===")
    print(f"Anzahl Vorhersagen: {summary['n_predictions']}")
    print(f"Durchschnittlicher Fehler: {summary['avg_error']:.4f}")
    print(f"Aktuelle Lernrate: {summary['current_learning_rate']:.6f}")
    print(f"Modell-Stabilität: {summary['model_stability']:.4f}")
    print(f"Regime-Wahrscheinlichkeiten: {[f'{p:.3f}' for p in summary['regime_probabilities']]}")

if __name__ == "__main__":
    demonstrate_model()

