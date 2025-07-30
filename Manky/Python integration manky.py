from minimal_data_predictor import MinimalDataPredictionSystem

# System initialisieren
symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
system = MinimalDataPredictionSystem(symbols)

# Vorhersagen machen
predictions = system.make_predictions()

# Ergebnisse verwenden
for symbol, prediction in predictions.items():
    print(f"{symbol}: {prediction.predicted_change*100:+.2f}% "
          f"(Konfidenz: {prediction.confidence:.1%})")
