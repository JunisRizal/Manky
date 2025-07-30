#!/usr/bin/env python3
"""
Test verfügbare APIs mit Kryptowährungssymbolen
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import json
from datetime import datetime

def test_crypto_symbols():
    """Teste verschiedene Kryptowährungssymbole"""
    client = ApiClient()
    
    # Verschiedene Krypto-Symbol-Formate testen
    crypto_symbols = [
        'BTC-USD',  # Bitcoin
        'ETH-USD',  # Ethereum
        'SOL-USD',  # Solana
        'ADA-USD',  # Cardano
        'DOGE-USD', # Dogecoin
        'MATIC-USD', # Polygon
        'AVAX-USD', # Avalanche
        'DOT-USD',  # Polkadot
        'LINK-USD', # Chainlink
        'UNI-USD'   # Uniswap
    ]
    
    successful_symbols = []
    failed_symbols = []
    
    print("=== Testing Cryptocurrency Symbols ===")
    print()
    
    for symbol in crypto_symbols:
        try:
            print(f"Testing {symbol}...")
            
            response = client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': '1d',
                'range': '7d',  # Last 7 days
                'includeAdjustedClose': True
            })
            
            if response and 'chart' in response and 'result' in response['chart']:
                result = response['chart']['result'][0]
                meta = result['meta']
                
                print(f"  ✓ Success: {symbol}")
                print(f"    Current Price: ${meta.get('regularMarketPrice', 'N/A')}")
                print(f"    Currency: {meta.get('currency', 'N/A')}")
                print(f"    Exchange: {meta.get('exchangeName', 'N/A')}")
                
                # Prüfe verfügbare Daten
                timestamps = result.get('timestamp', [])
                quotes = result.get('indicators', {}).get('quote', [{}])[0]
                
                print(f"    Data points: {len(timestamps)}")
                print(f"    Volume available: {'volume' in quotes and quotes['volume'] is not None}")
                print()
                
                successful_symbols.append({
                    'symbol': symbol,
                    'price': meta.get('regularMarketPrice'),
                    'currency': meta.get('currency'),
                    'exchange': meta.get('exchangeName'),
                    'data_points': len(timestamps),
                    'has_volume': 'volume' in quotes and quotes['volume'] is not None
                })
                
            else:
                print(f"  ✗ Failed: {symbol} - No data in response")
                failed_symbols.append(symbol)
                
        except Exception as e:
            print(f"  ✗ Failed: {symbol} - Error: {str(e)}")
            failed_symbols.append(symbol)
    
    print("=== Summary ===")
    print(f"Successful symbols: {len(successful_symbols)}")
    print(f"Failed symbols: {len(failed_symbols)}")
    print()
    
    if successful_symbols:
        print("Available cryptocurrency data:")
        for crypto in successful_symbols:
            print(f"  {crypto['symbol']}: ${crypto['price']} ({crypto['currency']}) - {crypto['data_points']} data points")
    
    if failed_symbols:
        print(f"Failed symbols: {', '.join(failed_symbols)}")
    
    return successful_symbols, failed_symbols

def test_detailed_crypto_data(symbol='BTC-USD'):
    """Teste detaillierte Daten für ein spezifisches Symbol"""
    client = ApiClient()
    
    print(f"\n=== Detailed Data Test for {symbol} ===")
    
    try:
        # Verschiedene Zeiträume testen
        intervals = ['1h', '1d']
        ranges = ['1d', '7d', '1mo']
        
        for interval in intervals:
            for range_period in ranges:
                print(f"\nTesting {interval} interval, {range_period} range...")
                
                response = client.call_api('YahooFinance/get_stock_chart', query={
                    'symbol': symbol,
                    'region': 'US',
                    'interval': interval,
                    'range': range_period,
                    'includeAdjustedClose': True
                })
                
                if response and 'chart' in response and 'result' in response['chart']:
                    result = response['chart']['result'][0]
                    timestamps = result.get('timestamp', [])
                    quotes = result.get('indicators', {}).get('quote', [{}])[0]
                    
                    print(f"  Data points: {len(timestamps)}")
                    
                    # Prüfe verfügbare Felder
                    available_fields = []
                    for field in ['open', 'high', 'low', 'close', 'volume']:
                        if field in quotes and quotes[field] is not None:
                            non_null_count = sum(1 for x in quotes[field] if x is not None)
                            available_fields.append(f"{field}({non_null_count})")
                    
                    print(f"  Available fields: {', '.join(available_fields)}")
                    
                    # Zeige letzte 3 Datenpunkte
                    if len(timestamps) >= 3:
                        print("  Last 3 data points:")
                        for i in range(-3, 0):
                            if i < len(timestamps):
                                date = datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d %H:%M')
                                close_price = quotes.get('close', [None] * len(timestamps))[i]
                                volume = quotes.get('volume', [None] * len(timestamps))[i]
                                print(f"    {date}: ${close_price}, Vol: {volume}")
                else:
                    print(f"  No data available for {interval}/{range_period}")
                    
    except Exception as e:
        print(f"Error testing detailed data: {str(e)}")

if __name__ == "__main__":
    # Teste verfügbare Kryptowährungen
    successful, failed = test_crypto_symbols()
    
    # Teste detaillierte Daten für Bitcoin
    if successful:
        test_detailed_crypto_data('BTC-USD')
    
    print("\n=== API Testing Complete ===")

