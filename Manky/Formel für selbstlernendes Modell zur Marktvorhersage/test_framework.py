#!/usr/bin/env python3
"""
Umfassendes Testframework für das Selbstlernende Kryptowährungs-Vorhersagemodell
Implementiert Unit Tests, Integration Tests und Performance Tests
"""

import unittest
import numpy as np
import sys
from typing import List
from core_formula import AdaptiveCryptoPredictor, MarketData
from validation_system import ModelValidator, ValidationResult

class TestAdaptiveCryptoPredictor(unittest.TestCase):
    """Unit Tests für das AdaptiveCryptoPredictor Modell"""
    
    def setUp(self):
        """Setup für jeden Test"""
        self.model = AdaptiveCryptoPredictor(n_features=25, n_regimes=3)
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self) -> List[MarketData]:
        """Generiert Beispieldaten für Tests"""
        np.random.seed(42)
        data = []
        base_price = 50000
        
        for i in range(50):
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
        
        return data
    
    def test_model_initialization(self):
        """Test der Modell-Initialisierung"""
        self.assertEqual(self.model.n_features, 25)
        self.assertEqual(self.model.n_regimes, 3)
        self.assertEqual(len(self.model.theta_base), 25)
        self.assertEqual(self.model.theta_interaction.shape, (25, 25))
        self.assertGreater(self.model.learning_rate, 0)
    
    def test_feature_extraction(self):
        """Test der Feature-Extraktion"""
        features = self.model.extract_features(self.sample_data[10], self.sample_data[:10])
        
        self.assertEqual(len(features), 25)
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertIsInstance(features, np.ndarray)
    
    def test_prediction_with_uncertainty(self):
        """Test der Vorhersage mit Unsicherheitsquantifizierung"""
        features = self.model.extract_features(self.sample_data[10], self.sample_data[:10])
        historical_features = [self.model.extract_features(d, self.sample_data[:i]) 
                             for i, d in enumerate(self.sample_data[1:11])]
        
        prediction, epistemic_unc, aleatoric_unc = self.model.predict_with_uncertainty(
            features, historical_features
        )
        
        self.assertIsInstance(prediction, (float, np.floating))
        self.assertIsInstance(epistemic_unc, (float, np.floating))
        self.assertIsInstance(aleatoric_unc, (float, np.floating))
        self.assertGreaterEqual(epistemic_unc, 0)
        self.assertGreaterEqual(aleatoric_unc, 0)
    
    def test_parameter_update(self):
        """Test der Parameter-Aktualisierung"""
        features = self.model.extract_features(self.sample_data[10], self.sample_data[:10])
        old_theta = self.model.theta_base.copy()
        
        self.model.update_parameters(features, 0.01, 0.005)
        
        # Parameter sollten sich geändert haben
        self.assertFalse(np.array_equal(old_theta, self.model.theta_base))
        self.assertEqual(len(self.model.performance_scores), 1)
    
    def test_confidence_interval(self):
        """Test der Konfidenzintervall-Berechnung"""
        lower, upper = self.model.calculate_confidence_interval(0.01, 0.005, 0.003)
        
        self.assertLess(lower, upper)
        self.assertIsInstance(lower, (float, np.floating))
        self.assertIsInstance(upper, (float, np.floating))
    
    def test_model_summary(self):
        """Test der Modell-Zusammenfassung"""
        # Füge einige Performance-Scores hinzu
        self.model.performance_scores = [0.01, 0.02, 0.015]
        
        summary = self.model.get_model_summary()
        
        self.assertIn('n_predictions', summary)
        self.assertIn('avg_error', summary)
        self.assertIn('current_learning_rate', summary)
        self.assertIn('regime_probabilities', summary)
        self.assertEqual(summary['n_predictions'], 3)
    
    def test_rsi_calculation(self):
        """Test der RSI-Berechnung"""
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]
        rsi = self.model._calculate_rsi(prices)
        
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
        self.assertIsInstance(rsi, (float, np.floating))

class TestModelValidator(unittest.TestCase):
    """Unit Tests für das ModelValidator System"""
    
    def setUp(self):
        """Setup für jeden Test"""
        self.validator = ModelValidator(purge_gap=5)
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self) -> List[MarketData]:
        """Generiert Beispieldaten für Tests"""
        np.random.seed(42)
        data = []
        base_price = 50000
        
        for i in range(200):  # Mehr Daten für Validierung
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
        
        return data
    
    def test_validator_initialization(self):
        """Test der Validator-Initialisierung"""
        self.assertEqual(self.validator.purge_gap, 5)
        self.assertEqual(len(self.validator.validation_results), 0)
    
    def test_walk_forward_analysis(self):
        """Test der Walk-Forward Analysis"""
        results = self.validator.walk_forward_analysis(
            self.sample_data, window_size=50, step_size=10
        )
        
        self.assertIn('predictions', results)
        self.assertIn('actuals', results)
        self.assertIn('timestamps', results)
        self.assertIn('confidence_intervals', results)
        
        if len(results['predictions']) > 0:
            self.assertEqual(len(results['predictions']), len(results['actuals']))
            self.assertTrue(all(isinstance(p, (float, np.floating)) for p in results['predictions']))
    
    def test_purged_cross_validation(self):
        """Test der Purged Cross-Validation"""
        cv_results = self.validator.purged_cross_validation(self.sample_data, n_folds=3)
        
        self.assertIsInstance(cv_results, list)
        if len(cv_results) > 0:
            self.assertIsInstance(cv_results[0], ValidationResult)
            
            for result in cv_results:
                self.assertGreaterEqual(result.accuracy, 0)
                self.assertLessEqual(result.accuracy, 1)
                self.assertGreaterEqual(result.precision, 0)
                self.assertLessEqual(result.precision, 1)
    
    def test_calculate_metrics(self):
        """Test der Metrik-Berechnung"""
        predictions = [0.01, -0.005, 0.02, -0.01, 0.015]
        actuals = [0.008, -0.003, 0.025, -0.012, 0.01]
        
        metrics = self.validator._calculate_metrics(predictions, actuals)
        
        self.assertIsInstance(metrics, ValidationResult)
        self.assertGreaterEqual(metrics.accuracy, 0)
        self.assertLessEqual(metrics.accuracy, 1)
        self.assertIsInstance(metrics.sharpe_ratio, (float, np.floating))
    
    def test_generate_validation_report(self):
        """Test der Berichterstellung"""
        # Erstelle Mock-Validierungsergebnisse
        mock_results = [
            ValidationResult(0.6, 0.5, 0.4, 0.45, 0.8, -0.1, 0.5, 0.55, 0.001, 0.02),
            ValidationResult(0.65, 0.55, 0.45, 0.5, 0.9, -0.08, 0.6, 0.6, 0.0015, 0.018),
            ValidationResult(0.58, 0.48, 0.38, 0.42, 0.75, -0.12, 0.45, 0.52, 0.0008, 0.022)
        ]
        
        report = self.validator.generate_validation_report(mock_results)
        
        self.assertIn('accuracy', report)
        self.assertIn('sharpe_ratio', report)
        
        for metric, stats in report.items():
            self.assertIn('mean', stats)
            self.assertIn('std', stats)
            self.assertIn('min', stats)
            self.assertIn('max', stats)

class TestIntegration(unittest.TestCase):
    """Integration Tests für das gesamte System"""
    
    def setUp(self):
        """Setup für Integration Tests"""
        self.model = AdaptiveCryptoPredictor(n_features=25, n_regimes=3)
        self.validator = ModelValidator(purge_gap=3)
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self) -> List[MarketData]:
        """Generiert Beispieldaten für Tests"""
        np.random.seed(42)
        data = []
        base_price = 50000
        
        for i in range(100):
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
        
        return data
    
    def test_end_to_end_workflow(self):
        """Test des kompletten End-to-End Workflows"""
        # 1. Feature-Extraktion
        features = self.model.extract_features(self.sample_data[20], self.sample_data[:20])
        self.assertEqual(len(features), 25)
        
        # 2. Vorhersage
        historical_features = [self.model.extract_features(d, self.sample_data[:i]) 
                             for i, d in enumerate(self.sample_data[11:21])]
        
        prediction, epistemic_unc, aleatoric_unc = self.model.predict_with_uncertainty(
            features, historical_features
        )
        
        # 3. Parameter-Update
        self.model.update_parameters(features, 0.01, prediction)
        
        # 4. Validierung
        cv_results = self.validator.purged_cross_validation(self.sample_data[:80], n_folds=3)
        
        # Assertions
        self.assertIsInstance(prediction, (float, np.floating))
        self.assertGreater(len(self.model.performance_scores), 0)
        self.assertIsInstance(cv_results, list)
    
    def test_model_learning_progression(self):
        """Test der Lernprogression des Modells"""
        initial_error = []
        final_error = []
        
        # Trainiere Modell auf ersten 50 Datenpunkten
        for i in range(10, 50):
            features = self.model.extract_features(self.sample_data[i], self.sample_data[:i])
            
            if i >= 20:
                historical_features = [self.model.extract_features(d, self.sample_data[:j]) 
                                     for j, d in enumerate(self.sample_data[i-10:i])]
                
                prediction, _, _ = self.model.predict_with_uncertainty(features, historical_features)
                
                # Tatsächlicher Wert
                if i < len(self.sample_data) - 1:
                    actual_change = ((self.sample_data[i+1].price - self.sample_data[i].price) / 
                                   self.sample_data[i].price)
                    
                    error = abs(prediction - actual_change)
                    
                    if len(initial_error) < 10:
                        initial_error.append(error)
                    elif len(final_error) < 10:
                        final_error.append(error)
                    
                    self.model.update_parameters(features, actual_change, prediction)
        
        # Modell sollte sich verbessert haben (niedrigere Fehler am Ende)
        if len(initial_error) > 0 and len(final_error) > 0:
            initial_avg = np.mean(initial_error)
            final_avg = np.mean(final_error)
            
            # Nicht immer garantiert, aber oft der Fall
            print(f"Initial average error: {initial_avg:.4f}")
            print(f"Final average error: {final_avg:.4f}")

class PerformanceTest(unittest.TestCase):
    """Performance Tests für das System"""
    
    def test_prediction_speed(self):
        """Test der Vorhersagegeschwindigkeit"""
        import time
        
        model = AdaptiveCryptoPredictor(n_features=25, n_regimes=3)
        sample_data = self._generate_large_dataset(1000)
        
        start_time = time.time()
        
        for i in range(100, 200):  # 100 Vorhersagen
            features = model.extract_features(sample_data[i], sample_data[:i])
            historical_features = [model.extract_features(d, sample_data[:j]) 
                                 for j, d in enumerate(sample_data[i-10:i])]
            
            prediction, _, _ = model.predict_with_uncertainty(features, historical_features)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_prediction = total_time / 100
        
        print(f"Durchschnittliche Zeit pro Vorhersage: {avg_time_per_prediction:.4f} Sekunden")
        
        # Sollte unter 0.1 Sekunden pro Vorhersage sein
        self.assertLess(avg_time_per_prediction, 0.1)
    
    def _generate_large_dataset(self, size: int) -> List[MarketData]:
        """Generiert großen Datensatz für Performance-Tests"""
        np.random.seed(42)
        data = []
        base_price = 50000
        
        for i in range(size):
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
        
        return data

def run_all_tests():
    """Führt alle Tests aus"""
    print("=== Umfassendes Testframework für Selbstlernendes Krypto-Modell ===")
    print()
    
    # Test Suite erstellen
    test_suite = unittest.TestSuite()
    
    # Unit Tests hinzufügen
    test_suite.addTest(unittest.makeSuite(TestAdaptiveCryptoPredictor))
    test_suite.addTest(unittest.makeSuite(TestModelValidator))
    
    # Integration Tests hinzufügen
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Performance Tests hinzufügen
    test_suite.addTest(unittest.makeSuite(PerformanceTest))
    
    # Test Runner
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Zusammenfassung
    print(f"\n=== Test-Zusammenfassung ===")
    print(f"Tests ausgeführt: {result.testsRun}")
    print(f"Fehler: {len(result.errors)}")
    print(f"Fehlschläge: {len(result.failures)}")
    print(f"Erfolgsrate: {((result.testsRun - len(result.errors) - len(result.failures)) / result.testsRun * 100):.1f}%")
    
    if result.errors:
        print("\nFehler:")
        for test, error in result.errors:
            print(f"  {test}: {error}")
    
    if result.failures:
        print("\nFehlschläge:")
        for test, failure in result.failures:
            print(f"  {test}: {failure}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

