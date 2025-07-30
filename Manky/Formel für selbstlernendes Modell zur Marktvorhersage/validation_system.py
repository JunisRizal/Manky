#!/usr/bin/env python3
"""
Validierungssystem für das Selbstlernende Kryptowährungs-Vorhersagemodell
Implementiert Walk-Forward Analysis, Purged Cross-Validation und Performance-Metriken
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from core_formula import AdaptiveCryptoPredictor, MarketData
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ValidationResult:
    """Struktur für Validierungsergebnisse"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    hit_rate: float
    avg_return: float
    volatility: float
    
class ModelValidator:
    """
    Umfassendes Validierungssystem für Kryptowährungs-Vorhersagemodelle
    """
    
    def __init__(self, purge_gap: int = 5):
        self.purge_gap = purge_gap
        self.validation_results = []
        
    def walk_forward_analysis(self, data: List[MarketData], 
                            window_size: int = 100,
                            step_size: int = 20,
                            prediction_horizon: int = 1) -> Dict:
        """
        Walk-Forward Analysis für realistische Backtesting
        """
        print("Führe Walk-Forward Analysis durch...")
        
        results = {
            'predictions': [],
            'actuals': [],
            'timestamps': [],
            'confidence_intervals': [],
            'model_states': []
        }
        
        # Initialisiere Modell
        model = AdaptiveCryptoPredictor(n_features=25, n_regimes=3)
        
        # Walk-Forward Schleife
        for start_idx in range(window_size, len(data) - prediction_horizon, step_size):
            end_idx = start_idx + window_size
            
            # Training auf historischen Daten
            training_data = data[start_idx-window_size:start_idx]
            test_data = data[start_idx:start_idx+prediction_horizon]
            
            # Modell trainieren
            self._train_model_on_window(model, training_data)
            
            # Vorhersage für Test-Periode
            if len(training_data) > 10:
                features = model.extract_features(training_data[-1], training_data[:-1])
                historical_features = [model.extract_features(d, training_data[:i]) 
                                     for i, d in enumerate(training_data[-10:])]
                
                prediction, epistemic_unc, aleatoric_unc = model.predict_with_uncertainty(
                    features, historical_features
                )
                
                # Konfidenzintervall
                lower, upper = model.calculate_confidence_interval(
                    prediction, epistemic_unc, aleatoric_unc
                )
                
                # Tatsächlicher Wert
                if start_idx + prediction_horizon < len(data):
                    actual_change = ((data[start_idx + prediction_horizon].price - 
                                    data[start_idx].price) / data[start_idx].price)
                    
                    results['predictions'].append(prediction)
                    results['actuals'].append(actual_change)
                    results['timestamps'].append(data[start_idx].timestamp)
                    results['confidence_intervals'].append((lower, upper))
                    results['model_states'].append(model.get_model_summary())
        
        return results
    
    def _train_model_on_window(self, model: AdaptiveCryptoPredictor, 
                              training_data: List[MarketData]):
        """Trainiert das Modell auf einem Datenfenster"""
        historical_features = []
        
        for i, market_data in enumerate(training_data[10:], 10):
            features = model.extract_features(market_data, training_data[:i])
            
            if i < len(training_data) - 1:
                # Berechne tatsächliche Änderung für Training
                next_price = training_data[i + 1].price
                actual_change = (next_price - market_data.price) / market_data.price
                
                # Vorhersage machen
                if len(historical_features) >= 10:
                    prediction, _, _ = model.predict_with_uncertainty(
                        features, historical_features[-10:]
                    )
                    
                    # Modell aktualisieren
                    model.update_parameters(features, actual_change, prediction)
            
            historical_features.append(features)
    
    def purged_cross_validation(self, data: List[MarketData], 
                               n_folds: int = 5) -> List[ValidationResult]:
        """
        Purged Cross-Validation zur Vermeidung von Data Leakage
        """
        print("Führe Purged Cross-Validation durch...")
        
        fold_size = len(data) // n_folds
        results = []
        
        for fold in range(n_folds):
            print(f"  Fold {fold + 1}/{n_folds}")
            
            # Test-Bereich definieren
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, len(data))
            
            # Training-Bereiche (mit Purge Gap)
            train_data = []
            
            # Daten vor Test-Bereich (mit Gap)
            if test_start > self.purge_gap:
                train_data.extend(data[:test_start - self.purge_gap])
            
            # Daten nach Test-Bereich (mit Gap)
            if test_end + self.purge_gap < len(data):
                train_data.extend(data[test_end + self.purge_gap:])
            
            # Test-Daten
            test_data = data[test_start:test_end]
            
            if len(train_data) > 50 and len(test_data) > 10:
                # Modell trainieren und validieren
                fold_result = self._validate_fold(train_data, test_data)
                results.append(fold_result)
        
        return results
    
    def _validate_fold(self, train_data: List[MarketData], 
                      test_data: List[MarketData]) -> ValidationResult:
        """Validiert ein einzelnes Fold"""
        model = AdaptiveCryptoPredictor(n_features=25, n_regimes=3)
        
        # Training
        self._train_model_on_window(model, train_data)
        
        # Testing
        predictions = []
        actuals = []
        
        for i, market_data in enumerate(test_data[:-1]):
            features = model.extract_features(market_data, train_data + test_data[:i])
            
            # Vorhersage
            if i >= 10:
                historical_features = [model.extract_features(d, train_data + test_data[:j]) 
                                     for j, d in enumerate(test_data[max(0, i-10):i])]
                
                prediction, _, _ = model.predict_with_uncertainty(
                    features, historical_features
                )
                
                # Tatsächlicher Wert
                next_price = test_data[i + 1].price
                actual_change = (next_price - market_data.price) / market_data.price
                
                predictions.append(prediction)
                actuals.append(actual_change)
        
        # Performance-Metriken berechnen
        return self._calculate_metrics(predictions, actuals)
    
    def _calculate_metrics(self, predictions: List[float], 
                          actuals: List[float]) -> ValidationResult:
        """Berechnet umfassende Performance-Metriken"""
        if not predictions or not actuals:
            return ValidationResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Accuracy (Richtungsgenauigkeit)
        direction_correct = np.sign(predictions) == np.sign(actuals)
        accuracy = np.mean(direction_correct)
        
        # Precision, Recall, F1 für positive Vorhersagen
        true_positives = np.sum((predictions > 0) & (actuals > 0))
        false_positives = np.sum((predictions > 0) & (actuals <= 0))
        false_negatives = np.sum((predictions <= 0) & (actuals > 0))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Trading-Performance simulieren
        returns = []
        positions = []
        
        for i, pred in enumerate(predictions):
            # Einfache Strategie: Long bei positiver Vorhersage
            position = 1 if pred > 0 else -1
            trade_return = position * actuals[i]
            
            returns.append(trade_return)
            positions.append(position)
        
        returns = np.array(returns)
        
        # Sharpe Ratio
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar Ratio
        calmar_ratio = avg_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Hit Rate
        hit_rate = np.mean(returns > 0)
        
        return ValidationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            hit_rate=hit_rate,
            avg_return=avg_return,
            volatility=volatility
        )
    
    def generate_validation_report(self, cv_results: List[ValidationResult]) -> Dict:
        """Generiert umfassenden Validierungsbericht"""
        if not cv_results:
            return {}
        
        metrics = {
            'accuracy': [r.accuracy for r in cv_results],
            'precision': [r.precision for r in cv_results],
            'recall': [r.recall for r in cv_results],
            'f1_score': [r.f1_score for r in cv_results],
            'sharpe_ratio': [r.sharpe_ratio for r in cv_results],
            'max_drawdown': [r.max_drawdown for r in cv_results],
            'calmar_ratio': [r.calmar_ratio for r in cv_results],
            'hit_rate': [r.hit_rate for r in cv_results],
            'avg_return': [r.avg_return for r in cv_results],
            'volatility': [r.volatility for r in cv_results]
        }
        
        report = {}
        for metric, values in metrics.items():
            report[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return report
    
    def plot_validation_results(self, wf_results: Dict, save_path: str = None):
        """Visualisiert Validierungsergebnisse"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Vorhersagen vs. Tatsächliche Werte
        axes[0, 0].scatter(wf_results['actuals'], wf_results['predictions'], alpha=0.6)
        axes[0, 0].plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='Perfect Prediction')
        axes[0, 0].set_xlabel('Tatsächliche Änderung')
        axes[0, 0].set_ylabel('Vorhergesagte Änderung')
        axes[0, 0].set_title('Vorhersagen vs. Tatsächliche Werte')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Zeitreihe der Vorhersagegenauigkeit
        if len(wf_results['predictions']) > 0:
            errors = np.abs(np.array(wf_results['predictions']) - np.array(wf_results['actuals']))
            window_size = min(20, len(errors) // 4)
            if window_size > 0:
                rolling_error = pd.Series(errors).rolling(window=window_size).mean()
                axes[0, 1].plot(rolling_error)
                axes[0, 1].set_xlabel('Zeit')
                axes[0, 1].set_ylabel('Rollierender Durchschnittsfehler')
                axes[0, 1].set_title(f'Vorhersagegenauigkeit über Zeit (Window: {window_size})')
                axes[0, 1].grid(True)
        
        # Verteilung der Vorhersagefehler
        if len(wf_results['predictions']) > 0:
            errors = np.array(wf_results['predictions']) - np.array(wf_results['actuals'])
            axes[1, 0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
            axes[1, 0].set_xlabel('Vorhersagefehler')
            axes[1, 0].set_ylabel('Häufigkeit')
            axes[1, 0].set_title('Verteilung der Vorhersagefehler')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Konfidenzintervall-Abdeckung
        if len(wf_results['confidence_intervals']) > 0:
            coverage = []
            for i, (actual, (lower, upper)) in enumerate(zip(wf_results['actuals'], 
                                                           wf_results['confidence_intervals'])):
                coverage.append(1 if lower <= actual <= upper else 0)
            
            window_size = min(20, len(coverage) // 4)
            if window_size > 0:
                rolling_coverage = pd.Series(coverage).rolling(window=window_size).mean()
                axes[1, 1].plot(rolling_coverage, label='Tatsächliche Abdeckung')
                axes[1, 1].axhline(0.95, color='red', linestyle='--', label='Erwartete Abdeckung (95%)')
                axes[1, 1].set_xlabel('Zeit')
                axes[1, 1].set_ylabel('Konfidenzintervall-Abdeckung')
                axes[1, 1].set_title('Kalibrierung der Unsicherheitsschätzung')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validierungsplot gespeichert: {save_path}")
        
        return fig

def demonstrate_validation():
    """Demonstriert das Validierungssystem"""
    print("=== Validierungssystem für Selbstlernendes Krypto-Modell ===")
    print()
    
    # Simulierte Daten generieren
    np.random.seed(42)
    data = []
    base_price = 50000
    
    print("Generiere Testdaten...")
    for i in range(500):  # Mehr Daten für robuste Validierung
        price_change = np.random.normal(0, 0.02)
        base_price *= (1 + price_change)
        
        market_data = MarketData(
            timestamp=i,
            price=base_price,
            volume=np.random.uniform(1e9, 5e9),
            open_interest=np.random.uniform(1e10, 3e11),
            funding_rate=np.random.normal(0.0001, 0.0005),
            long_short_ratio=np.random.uniform(0.8, 1.2),
            fear_greed_index=np.random.uniform(20, 80),
            sentiment_score=np.random.normal(0, 0.3),
            liquidations=np.random.uniform(1e8, 1e10),
            whale_alerts=np.random.poisson(5)
        )
        data.append(market_data)
    
    # Validator initialisieren
    validator = ModelValidator(purge_gap=5)
    
    # Walk-Forward Analysis
    print("Starte Walk-Forward Analysis...")
    wf_results = validator.walk_forward_analysis(data, window_size=100, step_size=20)
    
    print(f"Walk-Forward Ergebnisse:")
    print(f"  Anzahl Vorhersagen: {len(wf_results['predictions'])}")
    if len(wf_results['predictions']) > 0:
        accuracy = np.mean(np.sign(wf_results['predictions']) == np.sign(wf_results['actuals']))
        mae = np.mean(np.abs(np.array(wf_results['predictions']) - np.array(wf_results['actuals'])))
        print(f"  Richtungsgenauigkeit: {accuracy:.3f}")
        print(f"  Mittlerer absoluter Fehler: {mae:.4f}")
    print()
    
    # Purged Cross-Validation
    print("Starte Purged Cross-Validation...")
    cv_results = validator.purged_cross_validation(data, n_folds=5)
    
    if cv_results:
        report = validator.generate_validation_report(cv_results)
        
        print("Cross-Validation Ergebnisse:")
        for metric, stats in report.items():
            print(f"  {metric}:")
            print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print()
    
    # Visualisierung
    if len(wf_results['predictions']) > 0:
        print("Erstelle Validierungsvisualisierung...")
        fig = validator.plot_validation_results(wf_results, save_path='/home/ubuntu/validation_results.png')
        plt.close(fig)  # Schließe Figure um Speicher zu sparen
    
    print("Validierung abgeschlossen!")

if __name__ == "__main__":
    demonstrate_validation()

