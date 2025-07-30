import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import sqlite3
from pathlib import Path
import hashlib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@dataclass
class MarketData:
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    market_cap: float
    price_change_24h: float
    volume_change_24h: float
    
@dataclass
class WhaleAlert:
    timestamp: datetime
    symbol: str
    amount: float
    from_address: str
    to_address: str
    transaction_type: str
    usd_value: float

@dataclass
class NewsItem:
    timestamp: datetime
    title: str
    content: str
    source: str
    sentiment_score: float
    relevance_score: float
    symbols: List[str]

class BaseAgent(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(self.name)
        self.is_running = False
        self.data_buffer = []
        self.last_update = datetime.now()
        
    @abstractmethod
    async def collect_data(self) -> List[Any]:
        pass
    
    @abstractmethod
    def process_data(self, raw_data: Any) -> Any:
        pass
    
    def save_to_csv(self, data: List[Any], filename: str):
        """Save data to CSV with automatic backup"""
        try:
            df = pd.DataFrame([asdict(item) if hasattr(item, '__dict__') else item for item in data])
            
            # Create data directory if it doesn't exist
            Path("data").mkdir(exist_ok=True)
            
            filepath = f"data/{filename}"
            
            # Append to existing file or create new
            if Path(filepath).exists():
                df.to_csv(filepath, mode='a', header=False, index=False)
            else:
                df.to_csv(filepath, index=False)
                
            self.logger.info(f"Saved {len(data)} records to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")
    
    async def run(self):
        """Main agent loop"""
        self.is_running = True
        self.logger.info(f"Starting agent {self.name}")
        
        while self.is_running:
            try:
                # Collect data
                raw_data = await self.collect_data()
                
                if raw_data:
                    # Process data
                    processed_data = [self.process_data(item) for item in raw_data]
                    
                    # Add to buffer
                    self.data_buffer.extend(processed_data)
                    
                    # Save when buffer reaches threshold
                    if len(self.data_buffer) >= self.config.get('buffer_size', 100):
                        filename = f"{self.name.lower()}_{datetime.now().strftime('%Y%m%d')}.csv"
                        self.save_to_csv(self.data_buffer, filename)
                        self.data_buffer.clear()
                
                # Wait before next collection
                await asyncio.sleep(self.config.get('collection_interval', 60))
                
            except Exception as e:
                self.logger.error(f"Error in agent loop: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    def stop(self):
        self.is_running = False
        self.logger.info(f"Stopping agent {self.name}")

class PriceDataAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("PriceDataAgent", config)
        self.symbols = config.get('symbols', ['BTC', 'ETH', 'SOL', 'ADA', 'DOT'])
        self.api_url = "https://api.coingecko.com/api/v3/simple/price"
        
    async def collect_data(self) -> List[Dict]:
        """Collect live price data from CoinGecko API"""
        try:
            symbol_ids = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum', 
                'SOL': 'solana',
                'ADA': 'cardano',
                'DOT': 'polkadot'
            }
            
            ids = ','.join([symbol_ids.get(symbol, symbol.lower()) for symbol in self.symbols])
            
            params = {
                'ids': ids,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return list(data.items())
                    else:
                        self.logger.warning(f"API request failed with status {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error collecting price data: {e}")
            return []
    
    def process_data(self, raw_data: tuple) -> MarketData:
        """Process raw API data into MarketData objects"""
        coin_id, data = raw_data
        
        # Map coin_id back to symbol
        id_to_symbol = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'solana': 'SOL',
            'cardano': 'ADA',
            'polkadot': 'DOT'
        }
        
        symbol = id_to_symbol.get(coin_id, coin_id.upper())
        
        return MarketData(
            timestamp=datetime.now(),
            symbol=symbol,
            price=data.get('usd', 0),
            volume=data.get('usd_24h_vol', 0),
            market_cap=data.get('usd_market_cap', 0),
            price_change_24h=data.get('usd_24h_change', 0),
            volume_change_24h=0  # Not available in this API
        )

class WhaleAlertAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("WhaleAlertAgent", config)
        self.api_key = config.get('whale_alert_api_key', 'demo-key')
        self.api_url = "https://api.whale-alert.io/v1/transactions"
        self.min_value = config.get('min_whale_value', 1000000)  # $1M minimum
        
    async def collect_data(self) -> List[Dict]:
        """Collect whale transaction data"""
        try:
            # Get transactions from last hour
            start_time = int((datetime.now() - timedelta(hours=1)).timestamp())
            end_time = int(datetime.now().timestamp())
            
            params = {
                'api_key': self.api_key,
                'start': start_time,
                'end': end_time,
                'min_value': self.min_value
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('transactions', [])
                    else:
                        self.logger.warning(f"Whale Alert API request failed: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error collecting whale data: {e}")
            return []
    
    def process_data(self, raw_data: Dict) -> WhaleAlert:
        """Process raw whale alert data"""
        return WhaleAlert(
            timestamp=datetime.fromtimestamp(raw_data.get('timestamp', 0)),
            symbol=raw_data.get('symbol', ''),
            amount=raw_data.get('amount', 0),
            from_address=raw_data.get('from', {}).get('address', ''),
            to_address=raw_data.get('to', {}).get('address', ''),
            transaction_type=raw_data.get('transaction_type', ''),
            usd_value=raw_data.get('amount_usd', 0)
        )

class NewsAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("NewsAgent", config)
        self.api_key = config.get('news_api_key', 'demo-key')
        self.api_url = "https://newsapi.org/v2/everything"
        self.crypto_keywords = ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'DeFi']
        
    async def collect_data(self) -> List[Dict]:
        """Collect crypto-related news"""
        try:
            # Get news from last hour
            from_time = (datetime.now() - timedelta(hours=1)).isoformat()
            
            params = {
                'apiKey': self.api_key,
                'q': 'cryptocurrency OR bitcoin OR ethereum',
                'from': from_time,
                'sortBy': 'publishedAt',
                'language': 'en'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('articles', [])
                    else:
                        self.logger.warning(f"News API request failed: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error collecting news: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (can be replaced with more sophisticated NLP)"""
        positive_words = ['bullish', 'moon', 'pump', 'rise', 'gain', 'positive', 'bull', 'up']
        negative_words = ['bearish', 'crash', 'dump', 'fall', 'drop', 'negative', 'bear', 'down']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def extract_symbols(self, text: str) -> List[str]:
        """Extract crypto symbols from text"""
        symbols = []
        text_upper = text.upper()
        
        symbol_keywords = {
            'BITCOIN': 'BTC', 'BTC': 'BTC',
            'ETHEREUM': 'ETH', 'ETH': 'ETH',  
            'SOLANA': 'SOL', 'SOL': 'SOL',
            'CARDANO': 'ADA', 'ADA': 'ADA',
            'POLKADOT': 'DOT', 'DOT': 'DOT'
        }
        
        for keyword, symbol in symbol_keywords.items():
            if keyword in text_upper and symbol not in symbols:
                symbols.append(symbol)
        
        return symbols
    
    def process_data(self, raw_data: Dict) -> NewsItem:
        """Process raw news data"""
        title = raw_data.get('title', '')
        content = raw_data.get('description', '') or raw_data.get('content', '')
        full_text = f"{title} {content}"
        
        return NewsItem(
            timestamp=datetime.fromisoformat(raw_data.get('publishedAt', '').replace('Z', '+00:00')),
            title=title,
            content=content,
            source=raw_data.get('source', {}).get('name', ''),
            sentiment_score=self.analyze_sentiment(full_text),
            relevance_score=len(self.extract_symbols(full_text)) / 5.0,  # Normalize by max symbols
            symbols=self.extract_symbols(full_text)
        )

class AdaptiveLearningModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("AdaptiveLearningModel")
        self.models = {}  # One model per symbol
        self.scalers = {}  # Feature scalers per symbol
        self.feature_importance = {}
        self.performance_history = {}
        
        # Initialize SQLite database for structured storage
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for structured data storage"""
        self.db_path = "data/crypto_intelligence.db"
        Path("data").mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                predicted_price REAL,
                actual_price REAL,
                confidence REAL,
                features TEXT,
                regime TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                mse REAL,
                accuracy REAL,
                feature_importance TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_and_combine_data(self) -> Dict[str, pd.DataFrame]:
        """Load and combine data from all CSV files"""
        combined_data = {}
        
        try:
            # Load price data
            price_files = list(Path("data").glob("pricedataagent_*.csv"))
            if price_files:
                price_dfs = [pd.read_csv(f) for f in price_files]
                price_data = pd.concat(price_dfs, ignore_index=True)
                price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                
                for symbol in price_data['symbol'].unique():
                    combined_data[symbol] = price_data[price_data['symbol'] == symbol].copy()
            
            # Load whale data and merge
            whale_files = list(Path("data").glob("whalealertagent_*.csv"))
            if whale_files:
                whale_dfs = [pd.read_csv(f) for f in whale_files]
                whale_data = pd.concat(whale_dfs, ignore_index=True)
                whale_data['timestamp'] = pd.to_datetime(whale_data['timestamp'])
                
                # Aggregate whale activity per symbol per hour
                whale_agg = whale_data.groupby([
                    'symbol', 
                    pd.Grouper(key='timestamp', freq='H')
                ]).agg({
                    'usd_value': ['sum', 'count'],
                    'amount': 'sum'
                }).reset_index()
                
                whale_agg.columns = ['symbol', 'timestamp', 'whale_volume_usd', 'whale_count', 'whale_amount']
                
                # Merge with price data
                for symbol in combined_data.keys():
                    symbol_whales = whale_agg[whale_agg['symbol'] == symbol]
                    if not symbol_whales.empty:
                        combined_data[symbol] = pd.merge_asof(
                            combined_data[symbol].sort_values('timestamp'),
                            symbol_whales.sort_values('timestamp'),
                            on='timestamp',
                            suffixes=('', '_whale')
                        )
            
            # Load news data and create sentiment features
            news_files = list(Path("data").glob("newsagent_*.csv"))
            if news_files:
                news_dfs = [pd.read_csv(f) for f in news_files]
                news_data = pd.concat(news_dfs, ignore_index=True)
                news_data['timestamp'] = pd.to_datetime(news_data['timestamp'])
                
                # Process news sentiment per symbol
                for symbol in combined_data.keys():
                    # Filter news mentioning this symbol
                    symbol_news = news_data[news_data['symbols'].str.contains(symbol, na=False)]
                    
                    if not symbol_news.empty:
                        # Aggregate sentiment per hour
                        news_agg = symbol_news.groupby(
                            pd.Grouper(key='timestamp', freq='H')
                        ).agg({
                            'sentiment_score': 'mean',
                            'relevance_score': 'mean'
                        }).reset_index()
                        
                        news_agg['symbol'] = symbol
                        
                        # Merge with existing data
                        combined_data[symbol] = pd.merge_asof(
                            combined_data[symbol].sort_values('timestamp'),
                            news_agg.sort_values('timestamp'),
                            on='timestamp',
                            suffixes=('', '_news')
                        )
            
            self.logger.info(f"Loaded data for {len(combined_data)} symbols")
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return {}
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix from raw data"""
        features = df.copy()
        
        # Technical indicators
        features['price_sma_5'] = features['price'].rolling(5).mean()
        features['price_sma_20'] = features['price'].rolling(20).mean()
        features['price_momentum'] = features['price'].pct_change(5)
        features['volume_momentum'] = features['volume'].pct_change(5)
        
        # Volatility features
        features['price_volatility'] = features['price'].rolling(10).std()
        features['volume_volatility'] = features['volume'].rolling(10).std()
        
        # Whale activity features (fill NaN with 0)
        whale_cols = ['whale_volume_usd', 'whale_count', 'whale_amount']
        for col in whale_cols:
            if col in features.columns:
                features[col] = features[col].fillna(0)
            else:
                features[col] = 0
        
        # News sentiment features (fill NaN with 0)
        sentiment_cols = ['sentiment_score', 'relevance_score']
        for col in sentiment_cols:
            if col in features.columns:
                features[col] = features[col].fillna(0)
            else:
                features[col] = 0
        
        # Target variable (next hour price change)
        features['target'] = features['price'].shift(-1) / features['price'] - 1
        
        # Remove NaN rows
        features = features.dropna()
        
        return features
    
    def train_model(self, symbol: str, features_df: pd.DataFrame):
        """Train adaptive model for specific symbol"""
        if len(features_df) < 50:  # Minimum data requirement
            self.logger.warning(f"Insufficient data for {symbol}: {len(features_df)} rows")
            return
        
        try:
            # Prepare features and target
            feature_cols = [
                'price_sma_5', 'price_sma_20', 'price_momentum', 'volume_momentum',
                'price_volatility', 'volume_volatility', 'price_change_24h',
                'whale_volume_usd', 'whale_count', 'sentiment_score', 'relevance_score'
            ]
            
            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            X = features_df[feature_cols].values
            y = features_df['target'].values
            
            # Split data (80% train, 20% test)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            if symbol not in self.scalers:
                self.scalers[symbol] = StandardScaler()
            
            X_train_scaled = self.scalers[symbol].fit_transform(X_train)
            X_test_scaled = self.scalers[symbol].transform(X_test)
            
            # Train model
            if symbol not in self.models:
                self.models[symbol] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            
            self.models[symbol].fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.models[symbol].predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            
            # Calculate accuracy (within 1% prediction error)
            accuracy = np.mean(np.abs(y_test - y_pred) < 0.01)
            
            # Store feature importance
            self.feature_importance[symbol] = dict(zip(
                feature_cols, 
                self.models[symbol].feature_importances_
            ))
            
            # Update performance history
            if symbol not in self.performance_history:
                self.performance_history[symbol] = []
            
            self.performance_history[symbol].append({
                'timestamp': datetime.now(),
                'mse': mse,
                'accuracy': accuracy
            })
            
            # Store in database
            self.store_performance(symbol, mse, accuracy)
            
            self.logger.info(f"Model trained for {symbol}: MSE={mse:.6f}, Accuracy={accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {e}")
    
    def make_prediction(self, symbol: str, current_data: Dict) -> Optional[Dict]:
        """Make prediction for symbol using current data"""
        if symbol not in self.models or symbol not in self.scalers:
            return None
        
        try:
            # Prepare feature vector
            feature_vector = np.array([[
                current_data.get('price_sma_5', 0),
                current_data.get('price_sma_20', 0),
                current_data.get('price_momentum', 0),
                current_data.get('volume_momentum', 0),
                current_data.get('price_volatility', 0),
                current_data.get('volume_volatility', 0),
                current_data.get('price_change_24h', 0),
                current_data.get('whale_volume_usd', 0),
                current_data.get('whale_count', 0),
                current_data.get('sentiment_score', 0),
                current_data.get('relevance_score', 0)
            ]])
            
            # Scale features
            feature_vector_scaled = self.scalers[symbol].transform(feature_vector)
            
            # Make prediction
            predicted_change = self.models[symbol].predict(feature_vector_scaled)[0]
            current_price = current_data.get('price', 0)
            predicted_price = current_price * (1 + predicted_change)
            
            # Calculate confidence based on feature importance and recent performance
            confidence = self.calculate_confidence(symbol, current_data)
            
            prediction = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change': predicted_change,
                'confidence': confidence,
                'features': current_data
            }
            
            # Store prediction in database
            self.store_prediction(prediction)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error making prediction for {symbol}: {e}")
            return None
    
    def calculate_confidence(self, symbol: str, current_data: Dict) -> float:
        """Calculate prediction confidence based on various factors"""
        base_confidence = 0.5
        
        # Recent model performance
        if symbol in self.performance_history and self.performance_history[symbol]:
            recent_performance = self.performance_history[symbol][-5:]  # Last 5 evaluations
            avg_accuracy = np.mean([p['accuracy'] for p in recent_performance])
            base_confidence = avg_accuracy
        
        # Data quality factor
        data_quality = 0
        required_fields = ['price', 'volume', 'whale_volume_usd', 'sentiment_score']
        for field in required_fields:
            if current_data.get(field, 0) != 0:
                data_quality += 0.25
        
        # Feature importance factor
        if symbol in self.feature_importance:
            # Higher confidence if important features have strong signals
            importance_weight = 0
            for feature, importance in self.feature_importance[symbol].items():
                if current_data.get(feature, 0) != 0:
                    importance_weight += importance
            importance_weight = min(importance_weight, 1.0)
        else:
            importance_weight = 0.5
        
        # Combined confidence
        final_confidence = (
            base_confidence * 0.5 +
            data_quality * 0.3 +
            importance_weight * 0.2
        )
        
        return min(0.95, max(0.1, final_confidence))
    
    def store_prediction(self, prediction: Dict):
        """Store prediction in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (timestamp, symbol, predicted_price, confidence, features)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                prediction['timestamp'],
                prediction['symbol'],
                prediction['predicted_price'],
                prediction['confidence'],
                json.dumps(prediction['features'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing prediction: {e}")
    
    def store_performance(self, symbol: str, mse: float, accuracy: float):
        """Store model performance in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            importance_json = json.dumps(self.feature_importance.get(symbol, {}))
            
            cursor.execute('''
                INSERT INTO model_performance (timestamp, symbol, mse, accuracy, feature_importance)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now(), symbol, mse, accuracy, importance_json))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing performance: {e}")
    
    async def learning_loop(self):
        """Main learning loop - continuously retrain models"""
        self.logger.info("Starting adaptive learning loop")
        
        while True:
            try:
                # Load latest data
                combined_data = self.load_and_combine_data()
                
                # Train models for each symbol
                for symbol, data in combined_data.items():
                    if len(data) > 50:  # Minimum data requirement
                        features_df = self.create_features(data)
                        self.train_model(symbol, features_df)
                
                # Wait before next training cycle
                await asyncio.sleep(3600)  # Retrain every hour
                
            except Exception as e:
                self.logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

class AutonomousSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("AutonomousSystem")
        self.agents = []
        self.learning_model = AdaptiveLearningModel(config.get('model', {}))
        
        # Initialize agents
        self.init_agents()
    
    def init_agents(self):
        """Initialize all data collection agents"""
        # Price data agent
        price_config = self.config.get('price_agent', {
            'symbols': ['BTC', 'ETH', 'SOL', 'ADA', 'DOT'],
            'collection_interval': 300,  # 5 minutes
            'buffer_size': 50
        })
        self.agents.append(PriceDataAgent(price_config))
        
        # Whale alert agent
        whale_config = self.config.get('whale_agent', {
            'whale_alert_api_key': 'demo-key',
            'min_whale_value': 1000000,
            'collection_interval': 600,  # 10 minutes
            'buffer_size': 20
        })
        self.agents.append(WhaleAlertAgent(whale_config))
        
        # News agent
        news_config = self.config.get('news_agent', {
            'news_api_key': 'demo-key',
            'collection_interval': 900,  # 15 minutes
            'buffer_size': 30
        })
        self.agents.append(NewsAgent(news_config))
    
    async def run_system(self):
        """Run the complete autonomous system"""
        self.logger.info("Starting autonomous crypto intelligence system")
        
        # Start all agents
        agent_tasks = [agent.run() for agent in self.agents]
        
        # Start learning model
        learning_task = self.learning_model.learning_loop()
        
        # Run prediction loop
        prediction_task = self.prediction_loop()
        
        # Run all tasks concurrently
        await asyncio.gather(
            *agent_tasks,
            learning_task,
            prediction_task
        )
    
    async def prediction_loop(self):
        """Make predictions periodically"""
        self.logger.info("Starting prediction loop")
        
        # Wait for initial data collection and model training
        await asyncio.sleep(1800)  # Wait 30 minutes
        
        while True:
            try:
                # Load latest data for predictions
                combined_data = self.learning_model.load_and_combine_data()
                
                for symbol, data in combined_data.items():
                    if len(data) > 0:
                        # Get latest data point
                        latest_data = data.iloc[-1].to_dict()
                        
                        # Make prediction
                        prediction = self.learning_model.make_prediction(symbol, latest_data)