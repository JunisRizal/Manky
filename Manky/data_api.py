#!/usr/bin/env python3
"""
Data API Client for Crypto Prediction System
Provides access to external APIs for cryptocurrency data
"""

import requests
import time
import logging
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class ApiClient:
    """Simple API client for external data sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoPredictor/1.0'
        })
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
    def call_api(self, endpoint: str, query: Dict = None, method: str = 'GET') -> Optional[Dict[Any, Any]]:
        """
        Generic API call method
        For now, we'll use Yahoo Finance as a free alternative
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        try:
            if endpoint.startswith('YahooFinance/'):
                return self._call_yahoo_finance(endpoint, query)
            else:
                logger.warning(f"Unknown endpoint: {endpoint}")
                return None
                
        except Exception as e:
            logger.error(f"API call failed for {endpoint}: {str(e)}")
            return None
        finally:
            self.last_request_time = time.time()
    
    def _call_yahoo_finance(self, endpoint: str, query: Dict) -> Optional[Dict[Any, Any]]:
        """Call Yahoo Finance API (free alternative)"""
        symbol = query.get('symbol', 'BTC-USD')
        
        # Use Yahoo Finance API v8 (free)
        base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        url = f"{base_url}/{symbol}"
        
        params = {
            'region': query.get('region', 'US'),
            'interval': query.get('interval', '1h'),
            'range': query.get('range', '1d'),
            'includeAdjustedClose': query.get('includeAdjustedClose', True)
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'chart' in data and data['chart']['result']:
                return data
            else:
                logger.warning(f"No data returned for {symbol}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Yahoo Finance API error: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            data = self.call_api('YahooFinance/get_stock_chart', {
                'symbol': symbol,
                'interval': '1m',
                'range': '1d'
            })
            
            if data and 'chart' in data and data['chart']['result']:
                result = data['chart']['result'][0]
                meta = result.get('meta', {})
                return meta.get('regularMarketPrice')
                
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            
        return None

    def test_connection(self) -> bool:
        """Test if API connection is working"""
        try:
            price = self.get_current_price('BTC-USD')
            return price is not None
        except Exception:
            return False