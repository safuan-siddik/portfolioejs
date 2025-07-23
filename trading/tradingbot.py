"""
Advanced Trading Bot with Ensemble Learning and Advanced Features
================================================================
A comprehensive trading bot that uses ensemble learning with LSTM,
Neural Network, and XGBoost models, featuring advanced risk management,
hyperparameter optimization, walk-forward backtesting, and performance monitoring.

This module serves as the main entry point for the trading bot. It integrates
all other modules, including data management, feature engineering, model training,
and trading execution.

Key Components:
- MarketDataManager: Manages loading and saving of market data.
- TradingBot: The core class that orchestrates all trading activities.
- main: The main function that handles command-line arguments and runs the bot.

Available models:
- 'nn': Custom neural network
- 'lstm': Long Short-Term Memory network
- 'xgb': XGBoost regressor
- 'ensemble': Weighted ensemble of all models

Advanced Features:
- Hyperparameter optimization with Optuna
- Walk-forward backtesting
- Advanced risk metrics
- Dynamic ensemble weighting
- Enhanced feature engineering
- Model performance monitoring
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import logging
import os
import json
import argparse
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn.functional as F
import seaborn as sns
from matplotlib.dates import DateFormatter
import pickle

# Backtest utility functions moved inline
def save_backtest_summary(results, filename):
    """Save backtest results summary to file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        logging.error(f"Error saving backtest summary: {e}")

def print_backtest_report(results):
    """Print backtest results summary."""
    try:
        print("\n" + "="*50)
        print("BACKTEST RESULTS SUMMARY")
        print("="*50)
        print(f"Initial Capital: ${results.get('initial_capital', 0):,.2f}")
        print(f"Final Capital: ${results.get('final_capital', 0):,.2f}")
        print(f"Total Return: {results.get('total_return', 0)*100:.2f}%")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0)*100:.2f}%")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print(f"Win Rate: {results.get('win_rate', 0)*100:.2f}%")
        print("="*50)
    except Exception as e:
        logging.error(f"Error printing backtest report: {e}")
from sklearn.preprocessing import MinMaxScaler
from models import CustomNeuralNetwork, LSTMModel, train_model
# XGBoost model moved inline
class XGBoostModel:
    """XGBoost model for price prediction."""
    
    def __init__(self, **kwargs):
        self.model = None
        self.scaler = MinMaxScaler()
        self.params = kwargs
        
    def fit(self, X, y):
        """Train the XGBoost model."""
        try:
            import xgboost as xgb
            # Filter out non-XGBoost parameters
            xgb_params = {k: v for k, v in self.params.items() 
                         if k not in ['epochs', 'batch_size', 'learning_rate']}
            
            # Set default XGBoost parameters if not provided
            if 'n_estimators' not in xgb_params:
                xgb_params['n_estimators'] = 100
            if 'learning_rate' not in xgb_params:
                xgb_params['learning_rate'] = 0.1
            if 'max_depth' not in xgb_params:
                xgb_params['max_depth'] = 6
            if 'random_state' not in xgb_params:
                xgb_params['random_state'] = 42
                
            self.model = xgb.XGBRegressor(**xgb_params)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
        except ImportError:
            logging.warning("XGBoost not available, using Random Forest instead")
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
    
    def train(self, X, y, **kwargs):
        """Alias for fit method for compatibility."""
        # Ignore neural network specific parameters
        return self.fit(X, y)
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, filepath):
        """Save the model."""
        if self.model is not None:
            try:
                self.model.save_model(filepath)
            except:
                # Fallback to pickle if save_model not available
                with open(filepath, 'wb') as f:
                    pickle.dump(self.model, f)
    
    def load(self, filepath):
        """Load the model."""
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor()
            self.model.load_model(filepath)
        except Exception as e:
            try:
                # Fallback to pickle
                with open(filepath, 'rb') as f:
                    self.model = pickle.load(f)
            except Exception as e2:
                logging.error(f"Error loading XGBoost model: {e2}")
                self.model = None

# Import advanced modules
from hyperparameter_optimizer import HyperparameterOptimizer
from risk_metrics import RiskMetrics
from walk_forward_backtest import WalkForwardBacktest
from dynamic_ensemble import DynamicEnsemble
from feature_engineering import AdvancedFeatureEngineering
from model_monitoring import ModelPerformanceMonitor

# Set up logging
log_file = "trading_bot.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger = logging.getLogger("TradingBot")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
if '--optimize' in sys.argv:
    pass  # Do not add any handlers during optimization
else:
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Optional: fallback for console Unicode on Windows
import sys
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

class MarketDataManager:
    """
    Manages market data from local CSV files.

    This class is responsible for loading, saving, and validating market data
    from CSV files. It ensures that the data is in the correct format and
    handles date filtering.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the market data manager.

        Args:
            data_dir (str): The directory where the data files are stored.
        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Expected columns in CSV files
        self.required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    def load_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load market data from CSV files.

        This method loads all CSV files for a given symbol, combines them,
        and filters by date range.

        Args:
            symbol (str): The stock symbol to load data for.
            start_date (Optional[str]): The start date for the data range.
            end_date (Optional[str]): The end date for the data range.

        Returns:
            Optional[pd.DataFrame]: A DataFrame with the loaded data, or None if an error occurs.
        """
        try:
            # Get all data files for this symbol
            data_files = [f for f in os.listdir(self.data_dir) if f.startswith(f"{symbol}_") and f.endswith(".csv")]
            if not data_files:
                logger.error(f"No data files found for {symbol}")
                return None
            
            # Read and combine all data files
            dfs = []
            for file in data_files:
                file_path = os.path.join(self.data_dir, file)
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                dfs.append(df)
            
            # Combine all dataframes
            data = pd.concat(dfs, ignore_index=True)
            data = data.drop_duplicates(subset=['Date'])
            data = data.sort_values('Date')
            data.set_index('Date', inplace=True)
            
            # Filter by date range if provided
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]
            
            # Verify required columns
            missing_columns = [col for col in self.required_columns if col != 'Date' and col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns in data files: {missing_columns}")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def save_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Save market data to a CSV file.

        The filename is generated based on the symbol and the date range of the data.

        Args:
            symbol (str): The stock symbol.
            data (pd.DataFrame): The DataFrame to save.

        Returns:
            bool: True if the data was saved successfully, False otherwise.
        """
        try:
            # Create filename with date range
            start_date = data.index.min().strftime('%Y-%m-%d')
            end_date = data.index.max().strftime('%Y-%m-%d')
            filename = f"{symbol}_{start_date}_{end_date}.csv"
            file_path = os.path.join(self.data_dir, filename)
            
            # Save to CSV
            data.to_csv(file_path)
            logger.info(f"Data saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")
            return False

class TradingBot:
    """
    The core trading bot class.

    This class orchestrates all trading activities, including data fetching,
    model training, prediction, and trade execution. It integrates all other
    modules to create a complete trading system.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the trading bot with configuration parameters.

        Args:
            config_file (Optional[str]): Path to the configuration file.
        """
        # Default configuration
        self.config = {
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "lookback_days": 365,
            "test_size": 0.2,
            "feature_columns": ["Open", "High", "Low", "Close", "Volume"],
            "target_column": "Close",
            "sequence_length": 20,
            "training_epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.5,
            "hidden_size": 32,
            "model_type": "ensemble",  # Default to ensemble
            "initial_capital": 10000,
            "position_size": 0.25,  # Updated default
            "stop_loss_pct": 0.05,  # Updated default
            "take_profit_pct": 0.15,  # Updated default
            "trailing_stop_pct": 0.05,  # Updated default
            "max_holding_period": 45,  # Updated default
            "prediction_threshold": 0.001,  # Updated default
            "rsi_threshold": 50,  # Updated default
            "volume_threshold": 0.8,  # Updated default
            "volatility_threshold": 0.8,  # Updated default
            "models_dir": "models",
            "data_dir": "data",
            
            # Advanced features configuration
            "enable_hyperparameter_optimization": True,
            "enable_walk_forward_backtest": True,
            "enable_dynamic_ensemble": True,
            "enable_advanced_features": True,
            "enable_model_monitoring": True,
            
            # Ensemble configuration
            "ensemble_lookback": 30,
            "min_weight": 0.1,
            "max_weight": 0.6,
            "adaptation_rate": 0.1,
            
            # Monitoring configuration
            "alert_thresholds": {
                "error_increase": 0.5,
                "accuracy_drop": 0.1,
                "drift_threshold": 0.05,
                "consecutive_failures": 5
            },
            "short_window": 10,
            "medium_window": 30,
            "long_window": 100,
            "max_history": 1000
        }
        
        # Load configuration from file if provided, otherwise try to load config.json
        if config_file:
            config_path = config_file
        else:
            config_path = "config.json"
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    self.config.update(user_config)
                    print(f"Loaded configuration from {config_path}")
            else:
                print(f"Configuration file {config_path} not found, using defaults")
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
            print("Using default configuration")
        
        # Create directories if they don't exist
        directories = [
            self.config["models_dir"], 
            self.config["data_dir"],
            "walk_forward_results",
            "walk_forward_plots",
            "monitoring_data",
            "hyperparameter_results"
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize market data manager
        self.data_manager = MarketDataManager(self.config["data_dir"])
        
        # Initialize class variables
        self.models = {}
        self.scalers = {}
        self.data = {}
        self.portfolio = {
            "cash": self.config["initial_capital"],
            "positions": {},
            "history": []
        }
        
        self.logger = logging.getLogger("TradingBot")
        
        # Enhanced risk management parameters - use config values
        self.max_position_size = self.config.get("position_size", 0.25)
        self.stop_loss_pct = self.config.get("stop_loss_pct", 0.05)
        self.take_profit_pct = self.config.get("take_profit_pct", 0.15)
        self.max_drawdown = 0.15
        self.position_sizing_atr = 14
        self.max_positions = 3
        self.min_risk_reward = 2.0
        self.max_daily_trades = 2
        self.min_volume = 2000000
        
        # Strategy parameters
        self.trend_following = True
        self.volatility_filter = True
        self.momentum_threshold = 0.02
        self.volume_threshold = 1.5
        self.max_holding_period = 10
        self.min_profit_threshold = 0.01
        self.adaptive_position_sizing = True
        
        # Initialize advanced modules
        self._initialize_advanced_modules()
        
        logger.info(f"Advanced trading bot initialized with configuration: {self.config}")
        
        self.selected_features = {}
        self._load_selected_features_all_symbols()
    
    def _initialize_advanced_modules(self):
        """
        Initialize advanced modules for enhanced trading capabilities.

        This method initializes modules for dynamic ensemble, model monitoring,
        risk metrics, feature engineering, hyperparameter optimization, and
        walk-forward backtesting.
        """
        try:
            # Initialize dynamic ensemble
            if self.config.get("enable_dynamic_ensemble", True):
                from dynamic_ensemble import DynamicEnsemble
                self.dynamic_ensemble = DynamicEnsemble(self.config)
                logger.info("Dynamic ensemble initialized successfully")
            
            # Initialize model monitoring
            if self.config.get("enable_model_monitoring", True):
                from model_monitoring import ModelPerformanceMonitor
                self.model_monitor = ModelPerformanceMonitor(self.config)
                logger.info("Model monitor initialized successfully")
            
            # Initialize risk metrics
            from risk_metrics import RiskMetrics
            self.risk_metrics = RiskMetrics()
            logger.info("Risk metrics initialized successfully")
            
            # Initialize feature engineering
            if self.config.get("enable_advanced_features", True):
                from feature_engineering import AdvancedFeatureEngineering
                self.feature_engineering = AdvancedFeatureEngineering(self.config)
                logger.info("Feature engineering initialized successfully")
            
            # Initialize hyperparameter optimizer
            if self.config.get("enable_hyperparameter_optimization", True):
                from hyperparameter_optimizer import HyperparameterOptimizer
                self.hyperparameter_optimizer = HyperparameterOptimizer(
                    self.data_manager,
                    self.config
                )
                logger.info("Hyperparameter optimizer initialized successfully")
            
            # Initialize walk-forward backtest
            if self.config.get("enable_walk_forward_backtest", True):
                from walk_forward_backtest import WalkForwardBacktest
                self.walk_forward_backtest = WalkForwardBacktest(
                    self.config,
                    self._walkforward_data_preparer  # Pass the data preparation function
                )
                logger.info("Walk-forward backtest initialized successfully")
            
            logger.info("Advanced modules initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing advanced modules: {e}")
            # Continue without advanced modules
            pass
    
    def _walkforward_data_preparer(self, symbol, start_date, end_date):
        """
        Prepare raw data for walk-forward backtesting.

        This method combines data from multiple years to ensure sufficient data
        for rolling window backtests.

        Args:
            symbol (str): The stock symbol.
            start_date (str): The start date for the data.
            end_date (str): The end date for the data.

        Returns:
            Optional[pd.DataFrame]: A DataFrame with the prepared data, or None if an error occurs.
        """
        # For walk-forward backtest, we need more data than what's in a single year
        # Combine data from multiple years to get sufficient data for rolling windows
        try:
            # Get all available data files for this symbol
            data_files = [f for f in os.listdir(self.data_manager.data_dir) 
                         if f.startswith(f"{symbol}_") and f.endswith(".csv")]
            
            if not data_files:
                logger.error(f"No data files found for {symbol}")
                return None
            
            # Sort files by year to get chronological order
            data_files.sort()
            
            # Combine all available data
            combined_data = []
            for file_name in data_files:
                file_path = os.path.join(self.data_manager.data_dir, file_name)
                year_data = pd.read_csv(file_path)
                year_data['Date'] = pd.to_datetime(year_data['Date'])
                combined_data.append(year_data)
            
            if not combined_data:
                logger.error(f"Could not load any data for {symbol}")
                return None
            
            # Combine all years
            all_data = pd.concat(combined_data, ignore_index=True)
            all_data = all_data.sort_values('Date').reset_index(drop=True)
            all_data.set_index('Date', inplace=True)
            
            # Filter by date range if specified
            if start_date and end_date:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                all_data = all_data[(all_data.index >= start_dt) & (all_data.index <= end_dt)]
            
            # Ensure we have enough data for walk-forward backtest
            min_required_days = 315  # 252 training + 63 testing
            if len(all_data) < min_required_days:
                logger.warning(f"Insufficient data for walk-forward backtest: {len(all_data)} days (need at least {min_required_days})")
                return None
            
            logger.info(f"Prepared {len(all_data)} days of data for walk-forward backtest of {symbol}")
            return all_data
            
        except Exception as e:
            logger.error(f"Error preparing walk-forward data for {symbol}: {e}")
            return None
    
    def fetch_data(self, symbol, start_date=None, end_date=None):
        """
        Load market data for a given symbol from CSV files.

        This method serves as a wrapper around the MarketDataManager to fetch
        data for a specific symbol and date range.

        Args:
            symbol (str): The stock symbol.
            start_date (Optional[str]): The start date for the data.
            end_date (Optional[str]): The end date for the data.

        Returns:
            Optional[pd.DataFrame]: A DataFrame with the loaded data, or None if an error occurs.
        """
        if not start_date:
            # For training, use the entire available data
            start_date = "2022-01-01"
        if not end_date:
            end_date = "2024-12-31"
        
        logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
        
        # Load data from CSV
        data = self.data_manager.load_data(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            # Ensure we have enough data points for training
            min_required_points = self.config["sequence_length"] * 2  # At least 2 sequences
            if len(data) < min_required_points:
                logger.warning(f"Insufficient data points for {symbol}. Found {len(data)}, need at least {min_required_points}.")
                return None
            
            # Ensure all required columns are present and numeric
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"Missing required column {col} for {symbol}")
                    return None
                if not np.issubdtype(data[col].dtype, np.number):
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Drop any rows with NaN values
            data = data.dropna()
            
            if len(data) < min_required_points:
                logger.warning(f"Insufficient valid data points for {symbol} after cleaning. Found {len(data)}, need at least {min_required_points}.")
                return None
            
            return data
        else:
            logger.error(f"Could not load data for {symbol}")
            return None
        
    def fetch_latest_price(self, symbol):
        """
        Get the latest price for a symbol from the most recent CSV data.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Optional[float]: The latest price, or None if an error occurs.
        """
        try:
            # Get all data files for this symbol
            data_files = [f for f in os.listdir(self.data_manager.data_dir) if f.startswith(f"{symbol}_") and f.endswith(".csv")]
            if not data_files:
                logger.error(f"No data files found for {symbol}")
                return None
            
            # Sort files by date to get the most recent one
            data_files.sort(reverse=True)
            latest_file = data_files[0]
            logger.info(f"Using most recent data file for {symbol}: {latest_file}")
            
            # Read the latest file
            file_path = os.path.join(self.data_manager.data_dir, latest_file)
            data = pd.read_csv(file_path)
            
            # Validate data format
            if 'Date' not in data.columns or 'Close' not in data.columns:
                logger.error(f"Invalid data format in {latest_file}")
                return None
            
            # Convert date column to datetime
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Get the latest price
            latest_price = data['Close'].iloc[-1]
            
            # Validate price
            if pd.isna(latest_price) or latest_price <= 0:
                logger.error(f"Invalid price found for {symbol}: {latest_price}")
                return None
            
            logger.info(f"Latest price for {symbol}: ${latest_price:.2f}")
            return latest_price
            
        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            return None

    def prepare_data(self, symbol):
        """
        Prepare data for training and testing with advanced feature engineering.

        This method loads data, applies feature engineering, scales the data,
        and creates sequences for time series prediction.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
            X_train, X_test, y_train, and y_test tensors.
        """
        try:
            # Load data
            data = self.data_manager.load_data(symbol)
            if data is None or len(data) == 0:
                raise ValueError(f"No data available for {symbol}")
            
            # Use advanced feature engineering if enabled and we have enough data
            if hasattr(self, 'feature_engineering') and self.config.get("enable_advanced_features", True) and len(data) > 100:
                logger.info(f"Using advanced feature engineering for {symbol}")
                data = self.feature_engineering.create_all_features(data)
                
                # Check if we have enough data after feature engineering
                if len(data) > 50:
                    target = data['Close'].shift(-1).dropna()
                    feature_data = data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)[:-1]
                    
                    # Select top features (reduce number for smaller datasets)
                    n_features = min(30, len(feature_data.columns) // 2)
                    top_features = self.feature_engineering.select_top_features(
                        feature_data, target, n_features=n_features
                    )
                    feature_data = feature_data[top_features]
                    self.selected_features[symbol] = list(top_features)  # Store selected features
                    logger.info(f"Selected {len(top_features)} features for {symbol}")
                else:
                    # Use basic features for smaller datasets
                    logger.info(f"Switching to basic features for {symbol} (insufficient data after advanced feature engineering)")
                    feature_data = self._prepare_basic_features(data)
                    self.selected_features[symbol] = list(feature_data.columns)  # Store features
            else:
                # Fallback to basic feature preparation
                logger.info(f"Using basic feature engineering for {symbol} (insufficient data for advanced features)")
                feature_data = self._prepare_basic_features(data)
                self.selected_features[symbol] = list(feature_data.columns)  # Store features
            
            # Ensure all features are numeric
            for feature in feature_data.columns:
                if not np.issubdtype(feature_data[feature].dtype, np.number):
                    feature_data[feature] = pd.to_numeric(feature_data[feature], errors='coerce')
            
            # Drop any remaining NaN values after conversion
            feature_data = feature_data.dropna()
            
            if len(feature_data) < self.config["sequence_length"]:
                raise ValueError(f"Insufficient data points for {symbol}. Need at least {self.config['sequence_length']} points.")
            
            # Scale the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(feature_data.values)
            
            # Create sequences
            X, y = self.create_sequences(scaled_data, self.config["sequence_length"])
            
            # Defensive check: Ensure X and y are not empty
            if X.size == 0 or y.size == 0:
                raise ValueError(f"Sequence creation failed for {symbol}: X or y is empty. Data shape: {scaled_data.shape}")
            
            # Ensure data has correct shape for LSTM [batch_size, sequence_length, features]
            if len(X.shape) != 3:
                raise ValueError(f"X shape is not 3D after sequence creation for {symbol}: {X.shape}")
            
            # Split into train and test sets
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Convert to torch tensors
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.FloatTensor(y_train)
            y_test = torch.FloatTensor(y_test)
            
            # Log shapes for debugging
            logger.info(f"Advanced data prepared for {symbol}: X_train shape {X_train.shape}, y_train shape {y_train.shape}, Features={len(feature_data.columns)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error preparing data for {symbol}: {str(e)}")
            raise
    
    def _prepare_basic_features(self, data):
        """
        Prepare basic features as a fallback.

        This method is used when advanced feature engineering is disabled or
        when there is insufficient data.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: A DataFrame with basic features.
        """
        # Create features
        data['Returns'] = data['Close'].pct_change()
        data['Volume_Change'] = data['Volume'].pct_change()
        
        # Add technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])
        
        # Drop NaN values
        data = data.dropna()
        
        # Prepare features and target
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'Returns', 'Volume_Change', 'SMA_20', 'RSI', 
                   'MACD', 'MACD_Signal']
        
        return data[features]

    def create_sequences(self, data, seq_length):
        """
        Create sequences for time series prediction.

        Args:
            data (np.ndarray): The input data.
            seq_length (int): The length of each sequence.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the input sequences (X)
            and the target values (y).
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length), :])  # Use all features
            y.append(data[i + seq_length, 3])  # Target is 'Close', which is at index 3
        X = np.array(X)
        y = np.array(y)
        return X, y

    def calculate_rsi(self, prices, period=14):
        """
        Calculate the Relative Strength Index (RSI).

        Args:
            prices (pd.Series): A series of prices.
            period (int): The period for the RSI calculation.

        Returns:
            pd.Series: The RSI values.
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """
        Calculate the Moving Average Convergence Divergence (MACD).

        Args:
            prices (pd.Series): A series of prices.
            fast (int): The fast period for the EMA.
            slow (int): The slow period for the EMA.
            signal (int): The signal line period.

        Returns:
            Tuple[pd.Series, pd.Series]: A tuple containing the MACD line and the signal line.
        """
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """
        Calculate Bollinger Bands.

        Args:
            prices (pd.Series): Price series.
            period (int): Moving average period.
            std_dev (int): Standard deviation multiplier.

        Returns:
            tuple: (Upper band, Middle band, Lower band).
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def _detect_market_regime(self, data):
        """
        Detect market regime (bullish, bearish, neutral).

        Args:
            data (pd.DataFrame): Market data.

        Returns:
            str: Market regime ('bullish', 'bearish', 'neutral').
        """
        try:
            prices = data['Close']
            
            # Calculate moving averages
            sma_20 = prices.rolling(window=20).mean()
            sma_50 = prices.rolling(window=50).mean()
            sma_200 = prices.rolling(window=200).mean()
            
            # Calculate momentum
            momentum_20 = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20]
            momentum_50 = (prices.iloc[-1] - prices.iloc[-50]) / prices.iloc[-50]
            
            # Calculate volatility
            returns = prices.pct_change()
            volatility = returns.std() * np.sqrt(252)
            
            # Market regime logic
            current_price = prices.iloc[-1]
            
            # Bullish conditions
            bullish_conditions = [
                current_price > sma_20.iloc[-1],
                sma_20.iloc[-1] > sma_50.iloc[-1],
                sma_50.iloc[-1] > sma_200.iloc[-1],
                momentum_20 > 0.02,
                momentum_50 > 0.05,
                volatility < 0.4
            ]
            
            # Bearish conditions
            bearish_conditions = [
                current_price < sma_20.iloc[-1],
                sma_20.iloc[-1] < sma_50.iloc[-1],
                sma_50.iloc[-1] < sma_200.iloc[-1],
                momentum_20 < -0.02,
                momentum_50 < -0.05,
                volatility > 0.6
            ]
            
            bullish_score = sum(bullish_conditions)
            bearish_score = sum(bearish_conditions)
            
            if bullish_score >= 4:
                return 'bullish'
            elif bearish_score >= 4:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return 'neutral'
    
    def _load_optimized_parameters(self, symbol: str) -> Dict[str, Any]:
        """
        Load optimized hyperparameters for a symbol from a JSON file.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Dict[str, Any]: A dictionary of optimized parameters, or an empty dictionary if not found.
        """
        try:
            params_file = f"hyperparameter_results/{symbol}_optimized_params.json"
            if os.path.exists(params_file):
                with open(params_file, 'r', encoding='utf-8') as f:
                    optimized_params = json.load(f)
                logger.info(f"Loaded optimized parameters for {symbol}")
                return optimized_params
            else:
                logger.info(f"No optimized parameters found for {symbol}")
                return {}
        except Exception as e:
            logger.error(f"Error loading optimized parameters for {symbol}: {e}")
            return {}

    def train_model(self, symbol):
        """
        Train the model for a given symbol.

        This method prepares the data, initializes the model based on the
        configuration, and trains it using the optimized hyperparameters.

        Args:
            symbol (str): The stock symbol.

        Returns:
            bool: True if the model was trained successfully, False otherwise.
        """
        try:
            # Load optimized parameters if available
            optimized_params = self._load_optimized_parameters(symbol)
            
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(symbol)
            if X_train is None:
                return False
            # Defensive check: Ensure X_train is 3D and y_train is 1D or 2D
            logger.info(f"Training model for {symbol}: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
            if len(X_train.shape) != 3:
                raise ValueError(f"X_train shape is not 3D for {symbol}: {X_train.shape}")
            if y_train.size == 0:
                raise ValueError(f"y_train is empty for {symbol}")
            
            # Initialize model based on configuration
            if self.config["model_type"] == "lstm":
                # Use optimized parameters if available
                lstm_params = optimized_params.get("lstm", {})
                hidden_size = lstm_params.get("hidden_size", self.config["hidden_size"])
                sequence_length = lstm_params.get("sequence_length", self.config["sequence_length"])
                learning_rate = lstm_params.get("learning_rate", self.config["learning_rate"])
                batch_size = lstm_params.get("batch_size", self.config["batch_size"])
                num_layers = lstm_params.get("num_layers", 1)
                dropout = lstm_params.get("dropout", 0.2)
                
                input_size = X_train.shape[2]  # Number of features
                self.models[symbol] = LSTMModel(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout
                )
                
                # Train model with optimized parameters
                logger.info(f"Training LSTM model for {symbol} with optimized parameters")
                logger.info(f"Optimized params: hidden_size={hidden_size}, lr={learning_rate}, batch_size={batch_size}")
                history = train_model(
                    self.models[symbol], X_train, y_train,
                    epochs=self.config["training_epochs"],
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )
                
                # Evaluate model
                self.models[symbol].eval()
                with torch.no_grad():
                    y_pred = self.models[symbol](X_test)
                    mse = F.mse_loss(y_pred, y_test)
                    logger.info(f"Test MSE for {symbol}: {mse:.4f}")
                
                # Save all models and scalers
                self._save_models_and_scalers(symbol)
            elif self.config["model_type"] == "nn":
                # Use optimized parameters if available
                nn_params = optimized_params.get("nn", {})
                hidden_size = nn_params.get("hidden_size", self.config["hidden_size"])
                learning_rate = nn_params.get("learning_rate", self.config["learning_rate"])
                batch_size = nn_params.get("batch_size", self.config["batch_size"])
                
                input_size = X_train.shape[1] * X_train.shape[2]  # Flatten sequence and features
                self.models[symbol] = CustomNeuralNetwork(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    learning_rate=learning_rate
                )
                # Reshape data for simple neural network
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)
            
            # Train model with optimized parameters
                logger.info(f"Training neural network for {symbol} with optimized parameters")
                logger.info(f"Optimized params: hidden_size={hidden_size}, lr={learning_rate}, batch_size={batch_size}")
                history = self.models[symbol].train(
                    X_train, y_train,
                    epochs=self.config["training_epochs"],
                    batch_size=batch_size
                )
                
                # Evaluate model
                y_pred = self.models[symbol].predict(X_test)
                mse = np.mean((y_test.numpy() - y_pred) ** 2)
                logger.info(f"Test MSE for {symbol}: {mse:.4f}")
                
                # Save all models and scalers
                self._save_models_and_scalers(symbol)
            elif self.config["model_type"] == "xgb":
                # Use optimized parameters if available
                xgb_params = optimized_params.get("xgb", {})
                learning_rate = xgb_params.get("learning_rate", self.config["learning_rate"])
                n_estimators = xgb_params.get("n_estimators", 100)
                max_depth = xgb_params.get("max_depth", 6)
                subsample = xgb_params.get("subsample", 1.0)
                colsample_bytree = xgb_params.get("colsample_bytree", 1.0)
                min_child_weight = xgb_params.get("min_child_weight", 1)
                
                input_size = X_train.shape[1] * X_train.shape[2]  # Flatten sequence and features
                self.models[symbol] = XGBoostModel(
                    input_size=input_size,
                    hidden_size=self.config["hidden_size"],
                    learning_rate=learning_rate
                )
                
                # Apply optimized XGBoost parameters
                self.models[symbol].model.set_params(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight
                )
                
                # Reshape data for XGBoost
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)
                
                # Train model with optimized parameters
                logger.info(f"Training XGBoost model for {symbol} with optimized parameters")
                logger.info(f"Optimized params: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")
                history = self.models[symbol].train(
                    X_train, y_train,
                    epochs=self.config["training_epochs"],
                    batch_size=32
                )
                
                # Evaluate model
                y_pred = self.models[symbol].predict(X_test)
                mse = np.mean((y_test.numpy() - y_pred) ** 2)
                logger.info(f"Test MSE for {symbol}: {mse:.4f}")
                
                # Save all models and scalers
                self._save_models_and_scalers(symbol)
            
            # Save selected features after training
            self._save_selected_features(symbol)
            return True
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
            return False
    
    def load_model(self, symbol):
        """
        Load pre-trained models for a symbol.

        This method loads the LSTM, Neural Network, and XGBoost models, as well
        as the scaler, from disk.

        Args:
            symbol (str): The stock symbol.

        Returns:
            bool: True if the models were loaded successfully, False otherwise.
        """
        try:
            models_dir = self.config.get("models_dir", "models")
            
            # Load LSTM model
            lstm_path = os.path.join(models_dir, f"{symbol}_lstm.pth")
            if os.path.exists(lstm_path):
                try:
                    # Get data to determine input size
                    data = self.data_manager.load_data(symbol)
                    if data is not None:
                        # Use consistent feature engineering
                        if hasattr(self, 'feature_engineering') and self.config.get("enable_advanced_features", True) and symbol in self.selected_features and self.selected_features[symbol]:
                            data = self.feature_engineering.create_all_features(data)
                            selected_features = self.selected_features[symbol]
                            feature_data = data[selected_features]
                        else:
                            data['Returns'] = data['Close'].pct_change()
                            data['Volume_Change'] = data['Volume'].pct_change()
                            data['SMA_20'] = data['Close'].rolling(window=20).mean()
                            data['RSI'] = self.calculate_rsi(data['Close'])
                            data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])
                            data = data.dropna()
                            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Volume_Change', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal']
                            feature_data = data[features]
                        
                        input_size = len(feature_data.columns)
                        
                        self.models[symbol] = LSTMModel(
                            input_size=input_size,
                            hidden_size=self.config["hidden_size"]
                        )
                        self.models[symbol].load_weights(lstm_path)
                        logger.info(f"LSTM model loaded successfully for {symbol}")
                    else:
                        logger.warning(f"No data available for {symbol}, skipping LSTM model loading")
                except Exception as e:
                    logger.warning(f"Could not load LSTM model for {symbol}: {e}")
            
            # Load Neural Network model
            nn_path = os.path.join(models_dir, f"{symbol}_nn.pkl")
            if os.path.exists(nn_path):
                try:
                    # Get data to determine input size
                    data = self.data_manager.load_data(symbol)
                    if data is not None:
                        # Use consistent feature engineering
                        if hasattr(self, 'feature_engineering') and self.config.get("enable_advanced_features", True) and symbol in self.selected_features and self.selected_features[symbol]:
                            data = self.feature_engineering.create_all_features(data)
                            selected_features = self.selected_features[symbol]
                            feature_data = data[selected_features]
                        else:
                            data['Returns'] = data['Close'].pct_change()
                            data['Volume_Change'] = data['Volume'].pct_change()
                            data['SMA_20'] = data['Close'].rolling(window=20).mean()
                            data['RSI'] = self.calculate_rsi(data['Close'])
                            data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])
                            data = data.dropna()
                            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Volume_Change', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal']
                            feature_data = data[features]
                        
                        input_size = self.config["sequence_length"] * len(feature_data.columns)
                        
                        nn_model = CustomNeuralNetwork(
                            input_size=input_size,
                            hidden_size=self.config["hidden_size"],
                            output_size=1,
                            learning_rate=self.config["learning_rate"]
                        )
                        nn_model.load_weights(nn_path)
                        self.models[symbol+"_nn"] = nn_model
                        logger.info(f"Neural Network model loaded successfully for {symbol}")
                    else:
                        logger.warning(f"No data available for {symbol}, skipping NN model loading")
                except Exception as e:
                    logger.warning(f"Could not load Neural Network model for {symbol}: {e}")
            
            # Load XGBoost model
            xgb_path = os.path.join(models_dir, f"{symbol}_xgb.json")
            if os.path.exists(xgb_path):
                try:
                    # Get data to determine input size
                    data = self.data_manager.load_data(symbol)
                    if data is not None:
                        # Use consistent feature engineering
                        if hasattr(self, 'feature_engineering') and self.config.get("enable_advanced_features", True) and symbol in self.selected_features and self.selected_features[symbol]:
                            data = self.feature_engineering.create_all_features(data)
                            selected_features = self.selected_features[symbol]
                            feature_data = data[selected_features]
                        else:
                            data['Returns'] = data['Close'].pct_change()
                            data['Volume_Change'] = data['Volume'].pct_change()
                            data['SMA_20'] = data['Close'].rolling(window=20).mean()
                            data['RSI'] = self.calculate_rsi(data['Close'])
                            data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])
                            data = data.dropna()
                            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Volume_Change', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal']
                            feature_data = data[features]
                        
                        input_size = self.config["sequence_length"] * len(feature_data.columns)
                        
                        xgb_model = XGBoostModel(
                            input_size=input_size,
                            hidden_size=self.config["hidden_size"],
                            output_size=1,
                            learning_rate=self.config["learning_rate"]
                        )
                        xgb_model.load_weights(xgb_path)
                        self.models[symbol+"_xgb"] = xgb_model
                        logger.info(f"XGBoost model loaded successfully for {symbol}")
                    else:
                        logger.warning(f"No data available for {symbol}, skipping XGBoost model loading")
                except Exception as e:
                    logger.warning(f"Could not load XGBoost model for {symbol}: {e}")
            
            # Load scaler
            scaler_path = os.path.join(models_dir, f"{symbol}_scaler.pkl")
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as f:
                        self.scalers[symbol] = pickle.load(f)
                    logger.info(f"Scaler loaded successfully for {symbol}")
                except Exception as e:
                    logger.warning(f"Could not load scaler for {symbol}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models for {symbol}: {e}")
            return False
    
    def predict(self, symbol):
        """
        Make a prediction for the given symbol.

        This method prepares the latest data, makes a prediction using the
        trained model, and checks the entry conditions.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Tuple[Optional[float], List[bool]]: A tuple containing the predicted price
            and a list of booleans indicating which entry conditions were met.
        """
        try:
            # Get latest data
            data = self.data_manager.load_data(symbol)
            if data is None or len(data) == 0:
                raise ValueError(f"No data available for {symbol}")
            
            # Use the same feature engineering process as during training
            if hasattr(self, 'feature_engineering') and self.config.get("enable_advanced_features", True) and len(data) > 100:
                # Use advanced feature engineering (same as training)
                data = self.feature_engineering.create_all_features(data)
                
                # Check if we have enough data after feature engineering
                if len(data) > 50:
                    # Use selected features if available (same as training)
                    if symbol in self.selected_features and self.selected_features[symbol]:
                        features = self.selected_features[symbol]
                        # Check which features are actually available in the data
                        available_features = [f for f in features if f in data.columns]
                        missing_features = [f for f in features if f not in data.columns]
                        if missing_features:
                            logger.warning(f"Missing features for {symbol}: {missing_features}")
                        
                        # Use available features only (don't include OHLCV in features list)
                        data = data[available_features]
                        features = available_features
                    else:
                        # Use all engineered features except OHLCV
                        features = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
                        data = data[features]
                else:
                    # Use basic features for smaller datasets (same as training)
                    logger.info(f"Switching to basic features for {symbol} prediction (insufficient data after advanced feature engineering)")
                    data = self._prepare_basic_features(data)
                    features = list(data.columns)
            else:
                # Fallback to basic features (same as training)
                logger.info(f"Using basic feature engineering for {symbol} prediction")
                data = self._prepare_basic_features(data)
                features = list(data.columns)
            
            # Ensure all features are numeric
            for feature in features:
                if not np.issubdtype(data[feature].dtype, np.number):
                    data[feature] = pd.to_numeric(data[feature], errors='coerce')
            
            # Drop any remaining NaN values after conversion
            data = data.dropna()
            
            if len(data) < self.config["sequence_length"]:
                raise ValueError(f"Insufficient data points for {symbol}. Need at least {self.config['sequence_length']} points.")
            
            # Scale the data
            if symbol not in self.scalers:
                self.scalers[symbol] = MinMaxScaler()
                self.scalers[symbol].fit(data[features].values)
            
            scaled_data = self.scalers[symbol].transform(data[features].values)
            
            # Get the latest sequence
            latest_sequence = scaled_data[-self.config["sequence_length"]:]
            
            # Reshape for prediction [1, sequence_length, features]
            latest_sequence = latest_sequence.reshape(1, self.config["sequence_length"], -1)
            
            # Convert to tensor
            latest_sequence = torch.FloatTensor(latest_sequence)
            
            # Make prediction
            if symbol not in self.models:
                if not self.load_model(symbol):
                    self.logger.info(f"No model found for {symbol}. Training a new model...")
                    self.train_model(symbol)
                    if symbol not in self.models:
                        raise ValueError(f"Model for {symbol} could not be loaded or trained.")
            
            # Check if the model expects a different number of features
            expected_features = self.models[symbol].input_size if hasattr(self.models[symbol], 'input_size') else latest_sequence.shape[2]
            actual_features = latest_sequence.shape[2]
            
            if expected_features != actual_features:
                logger.warning(f"Feature mismatch for {symbol}: expected {expected_features}, got {actual_features}")
                # Try to pad or truncate features to match expected size
                if actual_features < expected_features:
                    # Pad with zeros
                    padding = torch.zeros(1, self.config["sequence_length"], expected_features - actual_features)
                    latest_sequence = torch.cat([latest_sequence, padding], dim=2)
                    logger.info(f"Padded features for {symbol} from {actual_features} to {expected_features}")
                elif actual_features > expected_features:
                    # Truncate to expected size
                    latest_sequence = latest_sequence[:, :, :expected_features]
                    logger.info(f"Truncated features for {symbol} from {actual_features} to {expected_features}")
            
            # Set model to evaluation mode
            self.models[symbol].eval()
            
            # Make prediction
            with torch.no_grad():
                prediction = self.models[symbol](latest_sequence)
                prediction = prediction.numpy()[0][0]  # Get scalar value
            
            # Inverse transform the prediction
            # Create a dummy array with the same shape as the original data
            dummy_array = np.zeros((1, len(features)))
            # Put the prediction in the Close price position (index 3)
            dummy_array[0, 3] = prediction
            # Inverse transform
            prediction = self.scalers[symbol].inverse_transform(dummy_array)[0, 3]
            
            # Get current price and indicators (with fallbacks)
            # We need to get the original data to access OHLCV columns
            original_data = self.data_manager.load_data(symbol)
            current_price = original_data['Close'].iloc[-1]
            
            # Try to get indicators, with fallbacks
            try:
                rsi = original_data['RSI'].iloc[-1] if 'RSI' in original_data.columns else 50
            except:
                rsi = 50
                
            try:
                macd = original_data['MACD'].iloc[-1] if 'MACD' in original_data.columns else 0
                signal = original_data['MACD_Signal'].iloc[-1] if 'MACD_Signal' in original_data.columns else 0
            except:
                macd, signal = 0, 0
                
            try:
                sma20 = original_data['SMA_20'].iloc[-1] if 'SMA_20' in original_data.columns else current_price
            except:
                sma20 = current_price
            
            # Log prediction and indicators
            logger.info(f"{original_data.index[-1]} - {symbol} - Current: {current_price:.2f}, Predicted: {prediction:.2f}, RSI: {rsi:.2f}, MACD: {macd:.2f}, Signal: {signal:.2f}, SMA20: {sma20:.2f}")
            
            # Check entry conditions
            entry_conditions = [
                prediction > current_price * 1.01,  # Predicted price is 1% higher
                rsi < 65,  # Not overbought
                macd > signal,  # MACD crossover
                current_price > sma20  # Price above SMA20
            ]
            
            # Log entry conditions
            logger.info(f"{symbol} entry conditions met: {sum(entry_conditions)}/4 | {entry_conditions}")
            
            return prediction, entry_conditions
            
        except Exception as e:
            self.logger.error(f"Error making prediction for {symbol}: {str(e)}")
            raise
    
    def calculate_position_size(self, symbol, current_price):
        """
        Calculate the position size based on advanced risk management and market conditions.

        This method uses volatility, trend strength, market regime, and Kelly Criterion
        to determine the optimal position size.

        Args:
            symbol (str): The stock symbol.
            current_price (float): The current price of the stock.

        Returns:
            int: The number of shares to trade.
        """
        try:
            # Get historical data for analysis
            data = self.data_manager.load_data(symbol)
            if data is None or data.empty:
                logger.error(f"Cannot calculate position size for {symbol}: No data available")
                return 0
            
            # Calculate volatility
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate trend strength using multiple timeframes
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=50).mean()
            sma_200 = data['Close'].rolling(window=200).mean()
            
            trend_strength_short = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
            trend_strength_long = (sma_50.iloc[-1] - sma_200.iloc[-1]) / sma_200.iloc[-1]
            
            # Calculate volume trend
            volume_sma = data['Volume'].rolling(window=20).mean()
            volume_trend = data['Volume'].iloc[-1] / volume_sma.iloc[-1]
            
            # Calculate RSI for momentum
            rsi = self.calculate_rsi(data['Close']).iloc[-1]
            
            # Calculate market regime
            market_regime = self._detect_market_regime(data)
            
            # ===== AGGRESSIVE RISK MANAGEMENT FOR BETTER RETURNS =====
            # Use risk metrics if available
            if hasattr(self, 'risk_metrics'):
                try:
                    portfolio_risk = self.risk_metrics.calculate_portfolio_risk(self.portfolio)
                    max_risk_per_trade = portfolio_risk.get('max_risk_per_trade', 0.08)  # MORE AGGRESSIVE
                    optimal_position_size = portfolio_risk.get('optimal_position_size', 0.08)  # MORE AGGRESSIVE
                except:
                    max_risk_per_trade = 0.08  # MORE AGGRESSIVE
                    optimal_position_size = 0.08  # MORE AGGRESSIVE
            else:
                max_risk_per_trade = 0.08  # MORE AGGRESSIVE
                optimal_position_size = 0.08  # MORE AGGRESSIVE
            
            # Dynamic risk adjustment based on market conditions
            if market_regime == 'bullish':
                risk_per_trade = min(max_risk_per_trade * 1.1, 0.06)  # Slight increase in bullish markets
            elif market_regime == 'bearish':
                risk_per_trade = max_risk_per_trade * 0.5  # Reduce risk in bearish markets
            else:
                risk_per_trade = max_risk_per_trade
            
            # Use Kelly Criterion for optimal position sizing
            win_rate = 0.5  # Default win rate
            avg_win = 0.15  # Default average win
            avg_loss = 0.05  # Default average loss
            
            if hasattr(self, 'model_monitor'):
                try:
                    historical_performance = self.model_monitor.get_historical_performance(symbol)
                    win_rate = historical_performance.get('win_rate', 0.5)
                    avg_win = historical_performance.get('avg_win', 0.15)
                    avg_loss = historical_performance.get('avg_loss', 0.05)
                except:
                    pass
            
            # Kelly Criterion calculation - MORE AGGRESSIVE for better returns
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% for better returns
            
            # Combine Kelly Criterion with risk management
            risk_per_trade = min(risk_per_trade, kelly_fraction)
            stop_loss_pct = self.config.get('stop_loss_pct', 0.05)
            base_position = self.portfolio['cash'] * risk_per_trade / stop_loss_pct
            
            # Volatility adjustment - much more aggressive for better performance
            volatility_factor = 1.0
            if volatility > 0.8:  # Very high volatility - still trade but reduce size
                volatility_factor = 0.7
            elif volatility > 0.6:  # High volatility - moderate reduction
                volatility_factor = 0.8
            elif volatility > 0.4:  # Medium volatility - slight boost
                volatility_factor = 1.2
            elif volatility < 0.2:  # Low volatility - significant boost
                volatility_factor = 1.6
            
            # Trend strength adjustment - much more aggressive
            trend_factor = 1.0
            if trend_strength_short > 0.02 and trend_strength_long > 0.04:  # Strong uptrend
                trend_factor = 1.8
            elif trend_strength_short > 0.005 and trend_strength_long > 0.015:  # Moderate uptrend
                trend_factor = 1.5
            elif trend_strength_short < -0.005 or trend_strength_long < -0.015:  # Downtrend
                trend_factor = 0.8
            
            # Volume adjustment - more aggressive
            volume_factor = 1.0
            if volume_trend > 1.5:  # High volume
                volume_factor = 1.3
            elif volume_trend > 1.0:  # Moderate volume
                volume_factor = 1.1
            elif volume_trend < 0.3:  # Low volume
                volume_factor = 0.8
            
            # RSI adjustment - more aggressive
            rsi_factor = 1.0
            if 25 < rsi < 75:  # Neutral RSI
                rsi_factor = 1.0
            elif rsi < 25:  # Oversold
                rsi_factor = 1.4
            elif rsi > 75:  # Overbought
                rsi_factor = 0.7
            
            # Market regime adjustment - more aggressive
            regime_factor = 1.0
            if market_regime == 'bullish':
                regime_factor = 1.4
            elif market_regime == 'bearish':
                regime_factor = 0.8
            elif market_regime == 'neutral':
                regime_factor = 1.1
            
            # Portfolio concentration adjustment - much more aggressive for better performance
            concentration_factor = 1.0
            total_positions = len(self.portfolio['positions'])
            if total_positions >= 10:  # Increased from 8 - allow even more positions
                concentration_factor = 0.9
            elif total_positions <= 2:  # Room for more positions
                concentration_factor = 1.5
            
            # Calculate final position size
            position_size = base_position * volatility_factor * trend_factor * volume_factor * rsi_factor * regime_factor * concentration_factor
            
            # Apply maximum position size limit - more conservative for better performance
            max_position_pct = self.config.get('position_size', 0.15)  # More conservative
            max_position = self.portfolio['cash'] * max_position_pct
            position_size = min(position_size, max_position)
            
            # Calculate number of shares
            shares = int(position_size / current_price)
            
            # Ensure minimum position size - much more aggressive for more trading opportunities
            min_position_value = 50  # Reduced from $100 to $50
            if shares * current_price < min_position_value:
                shares = 0
            
            # Log position sizing details
            logger.info(f"[DEBUG] Calculated position size for {symbol} on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}: {shares}")
            logger.info(f"  Base: ${base_position:.2f}, Vol: {volatility_factor:.2f}, Trend: {trend_factor:.2f}")
            logger.info(f"  Volume: {volume_factor:.2f}, RSI: {rsi_factor:.2f}, Regime: {regime_factor:.2f}")
            logger.info(f"  Final: {shares} shares (${shares * current_price:.2f})")
            
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    def check_stop_loss_take_profit(self, symbol, current_price):
        """
        Check if the stop-loss or take-profit levels have been hit.

        Args:
            symbol (str): The stock symbol.
            current_price (float): The current price of the stock.

        Returns:
            bool: True if a stop-loss or take-profit has been triggered, False otherwise.
        """
        if symbol not in self.portfolio['positions']:
            return False
            
        position = self.portfolio['positions'][symbol]
        entry_price = position['entry_price']
        shares = position['shares']
        
        # Calculate profit/loss percentage
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Check stop loss
        if pnl_pct <= -self.stop_loss_pct:
            logger.info(f"Stop loss triggered for {symbol} at {current_price:.2f}")
            return True
            
        # Check take profit
        if pnl_pct >= self.take_profit_pct:
            logger.info(f"Take profit triggered for {symbol} at {current_price:.2f}")
            return True
            
        return False
        
    def check_portfolio_risk(self):
        """
        Check and manage portfolio risk levels.

        This method checks for total exposure, individual position sizes,
        drawdown, and daily trading limits.

        Returns:
            bool: True if the portfolio risk is within acceptable limits, False otherwise.
        """
        try:
            # Calculate current portfolio value
            portfolio_value = self.portfolio['cash']
            for symbol, position in self.portfolio['positions'].items():
                current_price = self.fetch_latest_price(symbol)
                if current_price is not None:
                    portfolio_value += position['shares'] * current_price
            
            # Check if portfolio is empty
            if portfolio_value <= 0:
                logger.error("Portfolio value is zero or negative")
                return False
            
            # Calculate position weights
            position_weights = {}
            for symbol, position in self.portfolio['positions'].items():
                current_price = self.fetch_latest_price(symbol)
                if current_price is not None:
                    position_value = position['shares'] * current_price
                    position_weights[symbol] = position_value / portfolio_value
            
            # Check total exposure
            total_exposure = sum(position_weights.values())
            if total_exposure > 1.0:
                logger.warning(f"Total exposure ({total_exposure:.2%}) exceeds 100%")
                return False
            
            # Check individual position sizes
            for symbol, weight in position_weights.items():
                if weight > self.max_position_size:
                    logger.warning(f"Position in {symbol} ({weight:.2%}) exceeds maximum size")
                    return False
            
            # Check drawdown
            if hasattr(self, 'peak_portfolio_value'):
                current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
                if current_drawdown > self.max_drawdown:
                    logger.warning(f"Current drawdown ({current_drawdown:.2%}) exceeds maximum")
                    return False
            else:
                self.peak_portfolio_value = portfolio_value
            
            # Update peak value if current value is higher
            if portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = portfolio_value
            
            # Check daily trading limits
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            daily_trades = sum(1 for trade in self.portfolio['history'] 
                             if trade['timestamp'].startswith(today))
            
            if daily_trades >= self.max_daily_trades:
                logger.warning(f"Daily trading limit ({self.max_daily_trades}) reached")
                return False
            
            logger.info("Portfolio risk check passed")
            return True
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return False
        
    def execute_trade(self, symbol, action, shares=None):
        """
        Execute a trade with proper validation and risk management.

        This method handles the buying and selling of stocks, updates the
        portfolio, and records the trade.

        Args:
            symbol (str): The stock symbol.
            action (str): The action to take ('buy' or 'sell').
            shares (Optional[int]): The number of shares to trade.

        Returns:
            bool: True if the trade was executed successfully, False otherwise.
        """
        try:
            # Get current price
            current_price = self.fetch_latest_price(symbol)
            if current_price is None:
                logger.error(f"Cannot execute trade for {symbol}: Invalid price")
                return False
            
            # Validate action
            if action not in ['buy', 'sell']:
                logger.error(f"Invalid action: {action}")
                return False
            
            # Calculate position size if not provided
            if shares is None:
                position_value = self.portfolio['cash'] * self.config['position_size']
                shares = int(position_value / current_price)
            
            # Validate shares
            if shares <= 0:
                logger.error(f"Invalid number of shares: {shares}")
                return False
            
            # Calculate trade value
            trade_value = shares * current_price
            
            # Validate trade against portfolio constraints
            if action == 'buy':
                # Check if we have enough cash
                if trade_value > self.portfolio['cash']:
                    logger.warning(f"Insufficient cash for trade. Required: ${trade_value:.2f}, Available: ${self.portfolio['cash']:.2f}")
                    return False
                
                # Check position size limit
                if trade_value > self.portfolio['cash'] * self.max_position_size:
                    logger.warning(f"Trade exceeds maximum position size. Reducing shares.")
                    shares = int((self.portfolio['cash'] * self.max_position_size) / current_price)
                    trade_value = shares * current_price
                
                # Check maximum positions limit
                if len(self.portfolio['positions']) >= self.max_positions:
                    logger.warning(f"Maximum number of positions reached ({self.max_positions})")
                    return False
                
            elif action == 'sell':
                # Check if we have enough shares
                if symbol not in self.portfolio['positions']:
                    logger.error(f"No position found for {symbol}")
                    return False
                
                current_shares = self.portfolio['positions'][symbol]['shares']
                if shares > current_shares:
                    logger.warning(f"Attempting to sell more shares than owned. Reducing to {current_shares}")
                    shares = current_shares
                    trade_value = shares * current_price
            
            # Execute the trade
            if action == 'buy':
                self.portfolio['cash'] -= trade_value
                if symbol in self.portfolio['positions']:
                    # Update existing position
                    pos = self.portfolio['positions'][symbol]
                    total_shares = pos['shares'] + shares
                    total_cost = (pos['shares'] * pos['avg_price']) + trade_value
                    pos['shares'] = total_shares
                    pos['avg_price'] = total_cost / total_shares
                else:
                    # Create new position
                    self.portfolio['positions'][symbol] = {
                        'shares': shares,
                        'avg_price': current_price,
                        'entry_date': datetime.datetime.now().strftime('%Y-%m-%d')
                    }
            
            else:  # sell
                self.portfolio['cash'] += trade_value
                pos = self.portfolio['positions'][symbol]
                pos['shares'] -= shares
                if pos['shares'] == 0:
                    del self.portfolio['positions'][symbol]
            
            # Record trade
            trade_record = {
                'symbol': symbol,
                'action': action,
                'shares': shares,
                'price': current_price,
                'value': trade_value,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.portfolio['history'].append(trade_record)
            
            logger.info(f"Executed {action} trade for {symbol}: {shares} shares at ${current_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False
    
    def run(self):
        """
        The main trading loop.

        This method runs an infinite loop that continuously checks for trading
        opportunities, executes trades, and manages the portfolio.
        """
        logger.info("Starting trading bot...")
        
        while True:
            try:
                logger.info("Starting new trading cycle...")
                
                # Check portfolio risk
                if not self.check_portfolio_risk():
                    logger.warning("Trading paused due to portfolio risk limits")
                    time.sleep(300)  # Wait 5 minutes before checking again
                    continue
                
                for symbol in self.config["symbols"]:
                    # Get latest price and prediction
                    current_price = self.fetch_latest_price(symbol)
                    if current_price is None or current_price <= 0:
                        logger.warning(f"Invalid price for {symbol}")
                        continue
                    
                    prediction, entry_conditions = self.predict(symbol)
                    if prediction is None:
                        logger.warning(f"Could not get prediction for {symbol}")
                        continue
                    
                    # Get technical indicators
                    data = self.data_manager.load_data(symbol)
                    if data is None or len(data) < 20:
                        logger.warning(f"Insufficient data for {symbol}")
                        continue
                    
                    # Calculate technical indicators
                    rsi = self.calculate_rsi(data['Close']).iloc[-1]
                    macd, signal = self.calculate_macd(data['Close'])
                    macd_value = macd.iloc[-1]
                    signal_value = signal.iloc[-1]
                    sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
                    
                    # Check stop loss and take profit
                    if symbol in self.portfolio['positions']:
                        if self.check_stop_loss_take_profit(symbol, current_price):
                            self.execute_trade(symbol, 'sell')
                            continue
                    
                    # Enhanced trading logic
                    if symbol in self.portfolio['positions']:
                        # Exit conditions
                        exit_conditions = [
                            prediction < current_price * 0.99,  # 1% below current price
                            rsi > 70,  # Overbought
                            macd_value < signal_value,  # MACD bearish crossover
                            current_price < sma_20  # Price below 20-day SMA
                        ]
                        if sum(exit_conditions) >= 2:  # At least 2 conditions met
                            self.execute_trade(symbol, 'sell')
                    else:
                        # Entry conditions
                        entry_conditions = [
                            prediction > current_price * 1.002,  # 0.2% above current price
                            rsi < 45,  # Less strict RSI condition
                            macd_value > signal_value,  # MACD bullish crossover
                            current_price > sma_20  # Price above 20-day SMA
                        ]
                        
                        # Log entry conditions
                        logger.info(f"{symbol} entry conditions: {entry_conditions}")
                        logger.info(f"Prediction: {prediction:.2f}, Current: {current_price:.2f}, RSI: {rsi:.2f}, MACD: {macd_value:.2f}, Signal: {signal_value:.2f}, SMA20: {sma_20:.2f}")
                        
                        if sum(entry_conditions) >= 1:  # At least 1 condition met
                            # Calculate position size
                            shares = self.calculate_position_size(symbol, current_price)
                            if shares > 0:
                                self.execute_trade(symbol, 'buy', shares)
                            else:
                                logger.warning(f"Invalid position size calculated for {symbol}: {shares}")
                
                # Print portfolio status
                total_value = self.portfolio['cash']
                for symbol, position in self.portfolio['positions'].items():
                    current_price = self.fetch_latest_price(symbol)
                    if current_price is not None:
                        position_value = position['shares'] * current_price
                        total_value += position_value
                        logger.info(f"{symbol}: {position['shares']} shares at {current_price:.2f} = ${position_value:.2f}")
                logger.info(f"Total Portfolio Value: ${total_value:.2f}")
                
                # Wait for next cycle
                logger.info("Waiting for next trading cycle...")
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def visualize_backtest_results(self, backtest_results, save_dir="backtest_plots"):
        """
        Generate enhanced visualizations for backtest results.

        This method creates plots for portfolio performance, trade analysis,
        rolling metrics, and more.

        Args:
            backtest_results (Dict[str, Any]): The results of the backtest.
            save_dir (str): The directory to save the plots in.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.dates import DateFormatter
            import numpy as np
            
            # Create directory for plots if it doesn't exist
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_theme(style="whitegrid")
            
            # Prepare data
            trades = backtest_results['trades']
            if not trades:
                logger.warning("No trades to visualize")
                return
                
            # Convert trades to DataFrame for easier manipulation
            trades_df = pd.DataFrame(trades)
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            trades_df = trades_df.sort_values('date')
            
            # Calculate portfolio value over time
            portfolio_values = []
            current_value = backtest_results['initial_capital']
            dates = []
            
            for _, trade in trades_df.iterrows():
                if trade['action'] == 'buy':
                    current_value -= trade['shares'] * trade['price']
                else:
                    current_value += trade['shares'] * trade['price']
                portfolio_values.append(current_value)
                dates.append(trade['date'])
            
            # Convert to numpy arrays for calculations
            portfolio_values = np.array(portfolio_values)
            dates = np.array(dates)
            
            # 1. Portfolio Value Over Time with Drawdown
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1], sharex=True)
            
            # Portfolio value
            ax1.plot(dates, portfolio_values, label='Portfolio Value', linewidth=2, color='blue')
            ax1.set_title('Portfolio Performance Analysis', fontsize=14, pad=20)
            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            
            # Drawdown
            running_max = np.maximum.accumulate(portfolio_values)
            drawdown = (running_max - portfolio_values) / running_max
            ax2.fill_between(dates, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
            ax2.set_ylabel('Drawdown', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'portfolio_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Trade Distribution with Win/Loss Analysis
            plt.figure(figsize=(12, 6))
            trade_profits = [trade['pnl'] for trade in trades if 'pnl' in trade]
            if trade_profits:
                # Create subplot for distribution
                plt.subplot(1, 2, 1)
                sns.histplot(trade_profits, bins=30, kde=True)
                plt.title('Trade P&L Distribution', fontsize=12)
                plt.xlabel('Profit/Loss ($)', fontsize=10)
                plt.ylabel('Frequency', fontsize=10)
                
                # Create subplot for win/loss ratio
                plt.subplot(1, 2, 2)
                wins = sum(1 for p in trade_profits if p > 0)
                losses = sum(1 for p in trade_profits if p < 0)
                plt.pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%', colors=['green', 'red'])
                plt.title('Win/Loss Ratio', fontsize=12)
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'trade_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Rolling Returns and Volatility
            if len(portfolio_values) > 20:
                plt.figure(figsize=(12, 6))
                returns = pd.Series(portfolio_values).pct_change().dropna()
                rolling_returns = returns.rolling(window=20).mean() * 252  # Annualized
                rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
                
                # Ensure dates align with returns
                plot_dates = dates[1:][:len(returns)]
                
                plt.plot(plot_dates, rolling_returns, label='Rolling Returns (20d)', color='blue')
                plt.plot(plot_dates, rolling_vol, label='Rolling Volatility (20d)', color='red')
                plt.title('Rolling Returns and Volatility', fontsize=14, pad=20)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Annualized Rate', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=10)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'rolling_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. Monthly Returns Heatmap
            if trade_profits:
                plt.figure(figsize=(12, 8))
                monthly_returns = pd.Series(trade_profits, index=dates[-len(trade_profits):]).resample('ME').sum()
                if not monthly_returns.empty:
                    monthly_returns_df = pd.DataFrame({
                        'year': monthly_returns.index.year,
                        'month': monthly_returns.index.month,
                        'returns': monthly_returns.values
                    })
                    monthly_returns_pivot = monthly_returns_df.pivot_table(
                        index='year',
                        columns='month',
                        values='returns',
                        aggfunc='sum'
                    )
                    sns.heatmap(monthly_returns_pivot, annot=True, fmt='.0f', cmap='RdYlGn', center=0)
                    plt.title('Monthly Returns Heatmap', fontsize=14, pad=20)
                    plt.xlabel('Month', fontsize=12)
                    plt.ylabel('Year', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'monthly_returns.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. Risk Metrics Dashboard
            plt.figure(figsize=(12, 8))
            
            # Calculate total return from daily returns
            total_return = (backtest_results['daily_returns'][-1] - backtest_results['initial_capital']) / backtest_results['initial_capital']
            
            metrics = [
                ('Win Rate', f"{backtest_results['metrics']['winning_trades']/backtest_results['metrics']['total_trades']:.1%}"),
                ('Profit Factor', f"{backtest_results['metrics']['profit_factor']:.2f}"),
                ('Sharpe Ratio', f"{backtest_results['metrics']['risk_reward_ratio']:.2f}"),
                ('Max Drawdown', f"{backtest_results['metrics']['max_drawdown']:.1%}"),
                ('Avg Trade', f"${backtest_results['metrics']['avg_profit_per_trade']:.2f}"),
                ('Total Return', f"{total_return:.1%}")
            ]
            
            plt.axis('off')
            table = plt.table(
                cellText=[[m[0], m[1]] for m in metrics],
                loc='center',
                cellLoc='center',
                colWidths=[0.4, 0.4]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            plt.title('Key Performance Metrics', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'metrics_dashboard.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 6. Trade Scatter Plot
            if trade_profits:
                plt.figure(figsize=(12, 6))
                trade_dates = dates[-len(trade_profits):]
                plt.scatter(trade_dates, trade_profits, alpha=0.6, c=['green' if p > 0 else 'red' for p in trade_profits])
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
                plt.title('Trade P&L Scatter Plot', fontsize=14, pad=20)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Profit/Loss ($)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'trade_scatter.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Enhanced visualizations saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            logger.error(f"Error details: {str(e.__class__.__name__)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def print_backtest_summary(self, backtest_results):
        """
        Print a formatted summary of backtest results.

        Args:
            backtest_results (Dict[str, Any]): The results of the backtest.
        """
        try:
            # Calculate additional metrics
            total_trades = backtest_results['metrics']['total_trades']
            winning_trades = backtest_results['metrics']['winning_trades']
            losing_trades = backtest_results['metrics']['losing_trades']
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate total return from daily returns
            total_return = (backtest_results['daily_returns'][-1] - backtest_results['initial_capital']) / backtest_results['initial_capital']
            
            # Print header
            print("\n" + "="*80)
            print("BACKTEST RESULTS SUMMARY".center(80))
            print("="*80)
            
            # Portfolio Performance
            print("\nPORTFOLIO PERFORMANCE")
            print("-"*40)
            print(f"Initial Capital: ${backtest_results['initial_capital']:,.2f}")
            print(f"Final Capital: ${backtest_results['daily_returns'][-1]:,.2f}")
            print(f"Total Return: {total_return:.2%}")
            print(f"Annualized Return: {backtest_results['metrics']['annualized_return']:.2%}")
            print(f"Maximum Drawdown: {backtest_results['metrics']['max_drawdown']:.2%}")
            print(f"Recovery Factor: {backtest_results['metrics']['recovery_factor']:.2f}")
            
            # Risk Metrics
            print("\nRISK METRICS")
            print("-"*40)
            print(f"Sharpe Ratio: {backtest_results['metrics']['risk_reward_ratio']:.2f}")
            print(f"Sortino Ratio: {backtest_results['metrics']['sortino_ratio']:.2f}")
            print(f"Calmar Ratio: {backtest_results['metrics']['calmar_ratio']:.2f}")
            print(f"Volatility: {backtest_results['metrics']['volatility']:.2%}")
            
            # Trade Statistics
            print("\nTRADE STATISTICS")
            print("-"*40)
            print(f"Total Trades: {total_trades}")
            print(f"Winning Trades: {winning_trades}")
            print(f"Losing Trades: {losing_trades}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Average Profit per Trade: ${backtest_results['metrics']['avg_profit_per_trade']:,.2f}")
            print(f"Profit Factor: {backtest_results['metrics']['profit_factor']:.2f}")
            print(f"Expectancy: ${backtest_results['metrics']['expectancy']:,.2f}")
            print(f"Average Trade Duration: {backtest_results['metrics']['avg_trade_duration']:.1f} days")
            print(f"Max Consecutive Wins: {backtest_results['metrics']['max_consecutive_wins']}")
            print(f"Max Consecutive Losses: {backtest_results['metrics']['max_consecutive_losses']}")
            
            print("\n" + "="*80)
            
        except Exception as e:
            logger.error(f"Error printing backtest summary: {str(e)}")
    
    def backtest(self, start_date=None, end_date=None):
        """
        Run a backtest of the trading strategy with learning capabilities.

        This method simulates the trading strategy over a historical period
        and evaluates its performance.

        Args:
            start_date (Optional[str]): The start date for the backtest.
            end_date (Optional[str]): The end date for the backtest.

        Returns:
            Dict[str, Any]: A dictionary containing the backtest results.
        """
        print("[DEBUG] Starting backtest...")
        logger.info("[DEBUG] Entered backtest method.")
        if not start_date:
            start_date = "2023-01-01"  # Use historical date range
        if not end_date:
            end_date = "2024-12-31"  # Use historical date range
        
        # Create directories for backtest data and results
        backtest_data_dir = os.path.join(self.config["data_dir"], "backtest_data")
        backtest_results_dir = os.path.join(self.config["data_dir"], "backtest_results")
        os.makedirs(backtest_data_dir, exist_ok=True)
        os.makedirs(backtest_results_dir, exist_ok=True)
        
        # Generate unique backtest ID
        backtest_id = f"backtest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting backtest {backtest_id} from {start_date} to {end_date}")
        print(f"[DEBUG] Backtest ID: {backtest_id} from {start_date} to {end_date}")
        
        # Initialize backtest portfolio
        backtest_portfolio = {
            "backtest_id": backtest_id,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": self.config["initial_capital"],
            "final_capital": self.config["initial_capital"],
            "cash": self.config["initial_capital"],
            "positions": {},
            "trades": [],
            "daily_returns": [],
            "peak_value": self.config["initial_capital"],
            "metrics": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "avg_profit_per_trade": 0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "avg_trade_duration": 0,
                "profit_factor": 0,
                "expectancy": 0,
                "annualized_return": 0,
                "volatility": 0,
                "sortino_ratio": 0,
                "calmar_ratio": 0,
                "max_drawdown": 0,
                "avg_drawdown": 0,
                "recovery_factor": 0,
                "risk_reward_ratio": 0,
                "model_performance": {}  # Track model prediction accuracy
            }
        }
        
        # Load previous backtest results for learning
        previous_results = self._load_previous_backtest_results(backtest_results_dir)
        
        # Adjust strategy parameters based on previous results
        if previous_results:
            self._adjust_strategy_parameters(previous_results)
        
        # Load selected features for all symbols
        self._load_selected_features_all_symbols()
        
        # Get data for all symbols
        all_data = {}
        for symbol in self.config["symbols"]:
            data = self.fetch_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                # Force advanced feature engineering for backtesting to match training
                if hasattr(self, 'feature_engineering') and self.config.get("enable_advanced_features", True):
                    logger.info(f"Using advanced feature engineering for {symbol} backtest")
                    print(f"[DEBUG] Feature engineering module exists: {hasattr(self, 'feature_engineering')}")
                    print(f"[DEBUG] Advanced features enabled: {self.config.get('enable_advanced_features', True)}")
                    try:
                        # Use advanced feature engineering
                        data = self.feature_engineering.create_all_features(data)
                        print(f"[DEBUG] Advanced features created successfully for {symbol}. Data shape: {data.shape}")
                    except Exception as e:
                        logger.error(f"Error creating advanced features for {symbol}: {e}")
                        print(f"[DEBUG] Error creating advanced features: {e}")
                        # Fallback to basic features
                        data['Returns'] = data['Close'].pct_change()
                        data['Volume_Change'] = data['Volume'].pct_change()
                        data['SMA_20'] = data['Close'].rolling(window=20).mean()
                        data['RSI'] = self.calculate_rsi(data['Close'])
                        data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])
                    
                    # Use selected features if available
                    if symbol in self.selected_features and self.selected_features[symbol]:
                        features = self.selected_features[symbol]
                        # Keep OHLCV and selected features
                        available_features = ['Open', 'High', 'Low', 'Close', 'Volume'] + [f for f in features if f in data.columns]
                        missing_features = [f for f in features if f not in data.columns]
                        if missing_features:
                            logger.warning(f"Missing features for {symbol}: {missing_features}")
                        data = data[available_features]
                    else:
                        logger.warning(f"No selected features found for {symbol}, using all engineered features")
                        # Use all engineered features except OHLCV
                        engineered_features = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
                        data = data[['Open', 'High', 'Low', 'Close', 'Volume'] + engineered_features]
                else:
                    # Fallback to basic features only if advanced features are disabled
                    logger.info(f"Advanced features disabled, using basic features for {symbol} backtest")
                    data['Returns'] = data['Close'].pct_change()
                    data['Volume_Change'] = data['Volume'].pct_change()
                    data['SMA_20'] = data['Close'].rolling(window=20).mean()
                    data['RSI'] = self.calculate_rsi(data['Close'])
                    data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])
                
                data = data.dropna()
                all_data[symbol] = data
                logger.info(f"Loaded {len(data)} data points for {symbol}")
                print(f"[DEBUG] Loaded {len(data)} data points for {symbol}")
            else:
                logger.warning(f"No data loaded for {symbol} in backtest.")
                print(f"[DEBUG] No data loaded for {symbol} in backtest.")
        
        if not all_data:
            logger.error("No data available for backtest")
            print("[DEBUG] No data available for backtest. Exiting.")
            return
        
        # Get common date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        logger.info(f"Backtesting over {len(dates)} trading days")
        
        # Pre-calculate predictions for all symbols to avoid infinite loops
        print("[DEBUG] Pre-calculating predictions for all symbols...")
        symbol_predictions = {}
        for symbol in self.config["symbols"]:
            if symbol in all_data:
                print(f"[DEBUG] Calculating predictions for {symbol}...")
                try:
                    # Get the latest prediction for this symbol
                    prediction, individual_predictions = self.ensemble_predict(symbol)
                    if prediction is not None:
                        symbol_predictions[symbol] = {
                            'ensemble': prediction,
                            'individual': individual_predictions
                        }
                        print(f"[DEBUG] {symbol} prediction: {prediction:.4f}")
                    else:
                        print(f"[DEBUG] No prediction available for {symbol}")
                except Exception as e:
                    print(f"[DEBUG] Error calculating prediction for {symbol}: {e}")
                    symbol_predictions[symbol] = None
        
        print(f"[DEBUG] Pre-calculations complete. Predictions available for: {list(symbol_predictions.keys())}")
        
        # Track trade statistics and model performance
        trade_durations = []
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        trade_profits = []
        model_predictions = []
        actual_prices = []
        
        # Add progress tracking and safety limits
        total_dates = len(dates)
        processed_dates = 0
        max_iterations = min(total_dates, 1000)  # Limit to 1000 iterations max
        
        # --- ADDED: Portfolio history and drawdown tracking ---
        portfolio_history = []
        drawdown_data = []
        running_max = None
        monthly_returns_dict = {}
        last_month = None
        month_start_value = None
        # -----------------------------------------------------

        print(f"[DEBUG] Starting backtest with {total_dates} dates, max {max_iterations} iterations")
        
        for date in dates:
            processed_dates += 1
            
            # Safety check to prevent infinite loops
            if processed_dates > max_iterations:
                print(f"[WARNING] Reached maximum iterations ({max_iterations}), stopping backtest")
                break
                
            # Progress indicator every 100 dates
            if processed_dates % 100 == 0:
                print(f"[PROGRESS] Processed {processed_dates}/{total_dates} dates ({processed_dates/total_dates*100:.1f}%)")
            
            current_portfolio_value = backtest_portfolio['cash']
            total_value = current_portfolio_value
            
            # Check for stop loss and take profit on existing positions
            for symbol in list(backtest_portfolio['positions'].keys()):
                if symbol not in all_data:
                    logger.warning(f"Skipping {symbol} on {date}: no data available for this symbol in all_data (positions loop).")
                    continue
                position = backtest_portfolio['positions'][symbol]
                data = all_data[symbol]
                if date in data.index:
                    current_price = data.loc[date, 'Close']
                else:
                    prev_dates = data.index[data.index <= date]
                    if len(prev_dates) == 0:
                        continue
                    current_price = data.loc[prev_dates[-1], 'Close']
                
                # Use enhanced exit conditions for better performance
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                should_exit = self.check_exit_conditions(symbol, position, current_price)
                if should_exit:
                    # Execute sell
                    proceeds = position['shares'] * current_price
                    backtest_portfolio['cash'] += proceeds
                    trade_profit = proceeds - (position['shares'] * position['entry_price'])
                    trade_profits.append(trade_profit)
                    
                    backtest_portfolio['trades'].append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'sell',
                        'shares': position['shares'],
                        'price': current_price,
                        'pnl': trade_profit
                    })
                    
                    # Update trade statistics
                    backtest_portfolio['metrics']['total_trades'] += 1
                    if trade_profit > 0:
                        backtest_portfolio['metrics']['winning_trades'] += 1
                        consecutive_wins += 1
                        consecutive_losses = 0
                    else:
                        backtest_portfolio['metrics']['losing_trades'] += 1
                        consecutive_losses += 1
                        consecutive_wins = 0
                    
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    
                    # Calculate trade duration
                    duration = (pd.to_datetime(date) - pd.to_datetime(position['entry_date'])).days
                    trade_durations.append(duration)
                    
                    logger.info(f"SELL: {symbol} @ ${current_price:.2f} PnL: {pnl_pct:.2%}")
                    del backtest_portfolio['positions'][symbol]
            
            # Check for new trading opportunities
            for symbol in self.config["symbols"]:
                if symbol in backtest_portfolio['positions']:
                    continue  # Skip if we already have a position
                if symbol not in all_data:
                    logger.warning(f"Skipping {symbol} on {date}: no data available for this symbol in all_data (opportunities loop).")
                    continue
                data = all_data[symbol]
                if date not in data.index:
                    continue
                
                # Get current price and technical indicators (with fallbacks)
                current_price = data.loc[date, 'Close']
                
                # Try to get indicators, with fallbacks (using correct feature names from feature engineering)
                try:
                    rsi = data.loc[date, 'rsi_14'] if 'rsi_14' in data.columns else 50
                except:
                    rsi = 50
                    
                try:
                    macd_value = data.loc[date, 'macd'] if 'macd' in data.columns else 0
                    signal_value = data.loc[date, 'macd_signal'] if 'macd_signal' in data.columns else 0
                except:
                    macd_value, signal_value = 0, 0
                    
                try:
                    sma_20 = data.loc[date, 'sma_20'] if 'sma_20' in data.columns else current_price
                except:
                    sma_20 = current_price
                
                # Use pre-calculated ensemble prediction
                try:
                    # Use pre-calculated prediction to avoid infinite loops
                    if symbol not in symbol_predictions or symbol_predictions[symbol] is None:
                        logger.warning(f"No pre-calculated prediction available for {symbol} on {date}")
                        continue
                    
                    prediction = symbol_predictions[symbol]['ensemble']
                    model_predictions_dict = symbol_predictions[symbol]['individual']
                    
                    if prediction is None:
                        logger.warning(f"Pre-calculated prediction is None for {symbol} on {date}")
                        continue
                    
                    # Store prediction and actual price for learning
                    model_predictions.append(prediction)
                    actual_prices.append(current_price)
                    
                    # Log ensemble prediction details (reduced for speed)
                    logger.debug(f"[ENSEMBLE] {date} - {symbol} Prediction: {prediction:.4f}, Current: {current_price:.4f}")
                    
                    # Use enhanced entry conditions for better performance
                    should_enter = self.check_entry_conditions(symbol, data, prediction, current_price)
                    
                    # Execute trade if conditions are met
                    if should_enter:
                        # Calculate position size
                        shares = self.calculate_position_size(symbol, current_price)
                        logger.debug(f"[DEBUG] Calculated position size for {symbol} on {date}: {shares}")
                        if shares > 0 and shares * current_price <= backtest_portfolio['cash']:
                            # Execute buy
                            cost = shares * current_price
                            backtest_portfolio['cash'] -= cost
                            backtest_portfolio['positions'][symbol] = {
                                'shares': shares,
                                'entry_price': current_price,
                                'entry_date': date
                            }
                            backtest_portfolio['trades'].append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'buy',
                                'shares': shares,
                                'price': current_price
                            })
                            logger.info(f"BUY: {symbol} {shares} shares @ ${current_price:.2f}")
                        else:
                            logger.warning(f"[DEBUG] Trade skipped for {symbol} on {date}: Insufficient cash (${backtest_portfolio['cash']:.2f}) or shares ({shares}) for price ${current_price:.2f}")
                    else:
                        logger.debug(f"[DEBUG] No trade executed for {symbol} on {date}: Entry conditions not met")
                
                except Exception as e:
                    logger.error(f"Error processing {symbol} on {date}: {str(e)}")
                    continue
            
            # Check exit conditions for existing positions
            positions_to_remove = []
            for symbol, position in list(backtest_portfolio['positions'].items()):
                if symbol not in all_data:
                    logger.warning(f"Skipping {symbol} on {date}: no data available for this symbol in all_data (exit conditions loop).")
                    continue
                data = all_data[symbol]
                if date not in data.index:
                    continue
                
                current_price = data.loc[date, 'Close']
                entry_price = position['entry_price']
                entry_date = position['entry_date']
                shares = position['shares']
                
                # Calculate holding period
                holding_period = (pd.to_datetime(date) - pd.to_datetime(entry_date)).days
                
                # Calculate P&L
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Check exit conditions
                should_exit = self.check_exit_conditions(symbol, position, current_price, date)
                
                if should_exit:
                    # Execute sell
                    revenue = shares * current_price
                    backtest_portfolio['cash'] += revenue
                    
                    # Calculate profit/loss
                    profit = revenue - (shares * entry_price)
                    trade_profits.append(profit)
                    trade_durations.append(holding_period)
                    
                    # Record trade
                    backtest_portfolio['trades'].append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'sell',
                        'shares': shares,
                        'price': current_price,
                        'pnl': profit
                    })
                    
                    logger.info(f"SELL: {symbol} {shares} shares @ ${current_price:.2f} (P&L: ${profit:.2f})")
                    positions_to_remove.append(symbol)
            
            # Remove closed positions
            for symbol in positions_to_remove:
                del backtest_portfolio['positions'][symbol]
            
            # Calculate total portfolio value
            for symbol, position in backtest_portfolio['positions'].items():
                if symbol not in all_data:
                    logger.warning(f"Skipping {symbol} on {date}: no data available for this symbol in all_data (portfolio value loop).")
                    continue
                data = all_data[symbol]
                if date in data.index:
                    price = data.loc[date, 'Close']
                else:
                    prev_dates = data.index[data.index <= date]
                    if len(prev_dates) == 0:
                        continue
                    price = data.loc[prev_dates[-1], 'Close']
                total_value += position['shares'] * price

            # --- ADDED: Track portfolio value and drawdown ---
            portfolio_history.append({"date": str(date), "total_value": float(total_value)})
            if running_max is None or total_value > running_max:
                running_max = total_value
            drawdown = (running_max - total_value) / running_max if running_max > 0 else 0
            drawdown_data.append({"date": str(date), "drawdown": float(drawdown)})
            # Track monthly returns
            month = pd.to_datetime(date).strftime('%Y-%m')
            if last_month is None:
                last_month = month
                month_start_value = total_value
            if month != last_month:
                # Save return for last month
                if month_start_value is not None and month_start_value > 0:
                    monthly_return = (total_value - month_start_value) / month_start_value * 100
                    monthly_returns_dict[last_month] = monthly_return
                last_month = month
                month_start_value = total_value
            # ------------------------------------------------

            backtest_portfolio['daily_returns'].append(total_value)
            
            # Update peak value for drawdown calculation
            if total_value > backtest_portfolio['peak_value']:
                backtest_portfolio['peak_value'] = total_value
        
        # --- ADDED: Save last month's return ---
        if month_start_value is not None and last_month is not None and month_start_value > 0:
            monthly_return = (total_value - month_start_value) / month_start_value * 100
            monthly_returns_dict[last_month] = monthly_return
        # ---------------------------------------------------

        # --- ADDED: Save to backtest_portfolio for dashboard ---
        backtest_portfolio['portfolio_history'] = portfolio_history
        backtest_portfolio['drawdown_data'] = drawdown_data
        backtest_portfolio['monthly_returns'] = monthly_returns_dict
        # ------------------------------------------------------

        # Calculate model performance metrics
        if model_predictions and actual_prices:
            mse = np.mean((np.array(model_predictions) - np.array(actual_prices)) ** 2)
            mae = np.mean(np.abs(np.array(model_predictions) - np.array(actual_prices)))
            backtest_portfolio['metrics']['model_performance'] = {
                'mse': mse,
                'mae': mae,
                'prediction_accuracy': 1 - (mae / np.mean(actual_prices))
            }
        
        # Update final capital
        backtest_portfolio['final_capital'] = backtest_portfolio['daily_returns'][-1]
        
        # Calculate final metrics
        returns = pd.Series(backtest_portfolio['daily_returns']).pct_change().dropna()
        total_return = (backtest_portfolio['final_capital'] - backtest_portfolio['initial_capital']) / backtest_portfolio['initial_capital']
        
        # Update metrics
        if trade_profits:
            backtest_portfolio['metrics'].update({
                'avg_profit_per_trade': np.mean(trade_profits),
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
                'avg_trade_duration': np.mean(trade_durations),
                'profit_factor': abs(sum(p for p in trade_profits if p > 0) / sum(abs(p) for p in trade_profits if p < 0)) if sum(abs(p) for p in trade_profits if p < 0) != 0 else float('inf'),
                'expectancy': (sum(p for p in trade_profits if p > 0) / len(trade_profits)) - (sum(abs(p) for p in trade_profits if p < 0) / len(trade_profits)),
                'annualized_return': (1 + total_return) ** (252 / len(dates)) - 1,
                'volatility': returns.std() * np.sqrt(252),
                'sortino_ratio': (returns.mean() * np.sqrt(252)) / (returns[returns < 0].std() * np.sqrt(252)) if returns[returns < 0].std() != 0 else float('inf'),
                'calmar_ratio': (returns.mean() * np.sqrt(252)) / abs(returns.min()) if returns.min() != 0 else float('inf'),
                'max_drawdown': (np.maximum.accumulate(backtest_portfolio['daily_returns']) - backtest_portfolio['daily_returns']).max() / np.maximum.accumulate(backtest_portfolio['daily_returns']).max(),
                'avg_drawdown': (np.maximum.accumulate(backtest_portfolio['daily_returns']) - backtest_portfolio['daily_returns']).mean() / np.maximum.accumulate(backtest_portfolio['daily_returns']).max(),
                'recovery_factor': total_return / backtest_portfolio['metrics']['max_drawdown'] if backtest_portfolio['metrics']['max_drawdown'] != 0 else float('inf'),
                'risk_reward_ratio': abs(returns.mean() / returns.std()) if returns.std() != 0 else float('inf')
            })
        
        # Save backtest results
        results_file = os.path.join(backtest_results_dir, f"{backtest_id}_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(backtest_portfolio, f, indent=4, default=str)
        
        # Save backtest data
        data_file = os.path.join(backtest_data_dir, f"{backtest_id}_data.json")
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model_predictions': model_predictions,
                'actual_prices': actual_prices,
                'trade_profits': trade_profits,
                'trade_durations': trade_durations
            }, f, indent=4)
        
        logger.info(f"Backtest results saved to {results_file}")
        logger.info(f"Backtest data saved to {data_file}")
        
        # Skip visualizations for faster execution
        # self.visualize_backtest_results(backtest_portfolio, save_dir=os.path.join(backtest_results_dir, backtest_id))
        self.print_backtest_summary(backtest_portfolio)
        
        # Update model based on backtest results
        self._update_model_from_backtest(backtest_portfolio)
        
        return backtest_portfolio
    
    def _load_previous_backtest_results(self, results_dir):
        """
        Load previous backtest results for learning.

        Args:
            results_dir (str): The directory where the results are stored.

        Returns:
            List[Dict[str, Any]]: A list of previous backtest results.
        """
        try:
            results = []
            for file in os.listdir(results_dir):
                if file.endswith('_results.json'):
                    with open(os.path.join(results_dir, file), 'r', encoding='utf-8') as f:
                        results.append(json.load(f))
            return sorted(results, key=lambda x: x['backtest_id'])
        except Exception as e:
            logger.error(f"Error loading previous backtest results: {str(e)}")
            return []
    
    def _adjust_strategy_parameters(self, previous_results):
        """
        Adjust strategy parameters based on previous backtest results.

        This method uses the results of previous backtests to fine-tune the
        strategy parameters.

        Args:
            previous_results (List[Dict[str, Any]]): A list of previous backtest results.
        """
        try:
            if not previous_results:
                return
            
            # Get the most recent successful backtest
            recent_results = [r for r in previous_results if r['metrics']['profit_factor'] > 1.0]
            if not recent_results:
                return
            
            best_result = recent_results[-1]
            
            # Adjust entry conditions based on successful trades
            successful_trades = [t for t in best_result['trades'] if t.get('pnl', 0) > 0]
            if successful_trades:
                avg_profit = np.mean([t['pnl'] for t in successful_trades])
                self.take_profit_pct = min(0.1, max(0.02, avg_profit / best_result['initial_capital']))
                self.stop_loss_pct = min(0.05, max(0.01, self.take_profit_pct / 2))
            
            logger.info(f"Adjusted strategy parameters based on previous backtest {best_result['backtest_id']}")
            logger.info(f"New take profit: {self.take_profit_pct:.2%}, stop loss: {self.stop_loss_pct:.2%}")
            
        except Exception as e:
            logger.error(f"Error adjusting strategy parameters: {str(e)}")
    
    def _get_prediction_threshold(self, symbol):
        """
        Get a dynamic prediction threshold based on model performance.

        Args:
            symbol (str): The stock symbol.

        Returns:
            float: The prediction threshold.
        """
        try:
            if symbol in self.models:
                model_performance = getattr(self.models[symbol], 'performance', {})
                if model_performance:
                    accuracy = model_performance.get('accuracy', 0.5)
                    return max(0.001, min(0.02, 0.02 * (1 - accuracy)))
            return 0.002  # Default threshold
        except Exception as e:
            logger.error(f"Error getting prediction threshold: {str(e)}")
            return 0.002
    
    def _get_rsi_threshold(self, symbol):
        """
        Get a dynamic RSI threshold based on historical performance.

        Args:
            symbol (str): The stock symbol.

        Returns:
            float: The RSI threshold.
        """
        try:
            if symbol in self.models:
                model_performance = getattr(self.models[symbol], 'performance', {})
                if model_performance:
                    win_rate = model_performance.get('win_rate', 0.5)
                    return max(30, min(50, 40 + (win_rate - 0.5) * 20))
            return 45  # Default threshold
        except Exception as e:
            logger.error(f"Error getting RSI threshold: {str(e)}")
            return 45
    
    def _get_required_conditions(self, symbol):
        """
        Get the number of required conditions based on model confidence.

        Args:
            symbol (str): The stock symbol.

        Returns:
            int: The number of required conditions.
        """
        try:
            if symbol in self.models:
                model_performance = getattr(self.models[symbol], 'performance', {})
                if model_performance:
                    confidence = model_performance.get('confidence', 0.5)
                    return max(1, min(3, int(3 * (1 - confidence))))
            return 2  # Default requirement
        except Exception as e:
            logger.error(f"Error getting required conditions: {str(e)}")
            return 2
    
    def _update_model_from_backtest(self, backtest_results):
        """
        Update model parameters based on backtest results.

        This method updates the model's performance attributes based on the
        results of a backtest.

        Args:
            backtest_results (Dict[str, Any]): The results of the backtest.
        """
        try:
            for symbol in self.config["symbols"]:
                if symbol in self.models:
                    # Calculate model performance metrics
                    symbol_trades = [t for t in backtest_results['trades'] if t['symbol'] == symbol]
                    if symbol_trades:
                        win_rate = sum(1 for t in symbol_trades if t.get('pnl', 0) > 0) / len(symbol_trades)
                        avg_profit = np.mean([t.get('pnl', 0) for t in symbol_trades])
                        
                        # Update model performance attributes
                        self.models[symbol].performance = {
                            'win_rate': win_rate,
                            'avg_profit': avg_profit,
                            'confidence': min(1.0, max(0.0, win_rate * (1 + avg_profit/1000))),
                            'accuracy': backtest_results['metrics']['model_performance'].get('prediction_accuracy', 0.5)
                        }
                        
                        # Save updated model
                        model_path = os.path.join(self.config["models_dir"], f"{symbol}_{self.config['model_type']}.pth")
                        self.models[symbol].save_weights(model_path)
                        logger.info(f"Updated model for {symbol} with new performance metrics")
            
        except Exception as e:
            logger.error(f"Error updating model from backtest: {str(e)}")

    def check_entry_conditions(self, symbol, data, prediction, current_price):
        """
        Check the enhanced entry conditions using ALL advanced techniques for maximum performance.

        Args:
            symbol (str): The stock symbol.
            data (pd.DataFrame): The market data.
            prediction (float): The predicted price.
            current_price (float): The current price.

        Returns:
            bool: True if the entry conditions are met, False otherwise.
        """
        try:
            # ===== ADVANCED FEATURE ENGINEERING =====
            advanced_score = 0
            if hasattr(self, 'feature_engineering'):
                try:
                    advanced_features = self.feature_engineering.extract_features(data)
                    advanced_score = advanced_features.get('technical_score', 0) + \
                                   advanced_features.get('momentum_score', 0) + \
                                   advanced_features.get('volatility_score', 0)
                except:
                    pass
            
            # ===== DYNAMIC ENSEMBLE PREDICTION =====
            ensemble_confidence = 0.5
            if hasattr(self, 'dynamic_ensemble'):
                try:
                    ensemble_result = self.dynamic_ensemble.predict(symbol, data)
                    ensemble_confidence = ensemble_result.get('confidence', 0.5)
                    prediction = ensemble_result.get('prediction', prediction)
                except:
                    pass
            
            # ===== MODEL MONITORING RELIABILITY =====
            model_reliability = 0.5
            if hasattr(self, 'model_monitor'):
                try:
                    model_reliability = self.model_monitor.get_model_reliability(symbol)
                except:
                    pass
            
            # ===== HYPERPARAMETER OPTIMIZATION =====
            optimized_params = {}
            if hasattr(self, 'hyperparameter_optimizer'):
                try:
                    optimized_params = self.hyperparameter_optimizer.get_optimized_parameters(symbol)
                except:
                    pass
            
            # ===== ADVANCED TECHNICAL INDICATORS =====
            rsi = self.calculate_rsi(data['Close']).iloc[-1]
            macd, signal = self.calculate_macd(data['Close'])
            macd_value = macd.iloc[-1]
            signal_value = signal.iloc[-1]
            
            # Multiple timeframe moving averages
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            sma_200 = data['Close'].rolling(window=200).mean().iloc[-1]
            
            # Advanced momentum indicators
            momentum_5 = (current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
            momentum_10 = (current_price - data['Close'].iloc[-10]) / data['Close'].iloc[-10]
            momentum_20 = (current_price - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
            
            # Advanced volume analysis
            volume_sma = data['Volume'].rolling(window=20).mean()
            volume_trend = data['Volume'].iloc[-1] / volume_sma.iloc[-1]
            volume_ratio = data['Volume'].iloc[-5:].mean() / data['Volume'].iloc[-20:].mean()
            
            # Advanced volatility analysis
            returns = data['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)
            volatility_20 = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
            
            # Advanced Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['Close'])
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
            
            # Market regime detection
            market_trend = self._detect_market_regime(data)
            
            # ===== OPTIMIZED ENTRY CONDITIONS FOR MAXIMUM PERFORMANCE =====
            conditions = {
                # Core prediction conditions - OPTIMIZED FOR BETTER RETURNS
                'prediction': prediction > current_price * (1 + self.config.get('prediction_threshold', 0.001)),  # MUCH LOWER threshold
                'ensemble_confidence': ensemble_confidence > 0.3,  # MUCH LOWER confidence requirement
                'model_reliability': model_reliability > 0.3,  # MUCH LOWER reliability requirement
                
                # Technical indicators - OPTIMIZED FOR BETTER RETURNS
                'rsi': rsi < 75 and rsi > 20,  # MUCH MORE relaxed RSI range
                'macd': macd_value > signal_value * 0.99,  # MUCH LOWER MACD requirement
                'trend': current_price > sma_20 * 0.995,  # MUCH LOWER trend requirement
                
                # Momentum conditions - OPTIMIZED FOR BETTER RETURNS
                'momentum_5': momentum_5 > 0.001,  # MUCH LOWER momentum requirement
                'momentum_10': momentum_10 > 0.002,  # MUCH LOWER momentum requirement
                'momentum_20': momentum_20 > 0.003,  # MUCH LOWER momentum requirement
                
                # Volume conditions - OPTIMIZED FOR BETTER RETURNS
                'volume_trend': volume_trend > 0.4,  # MUCH LOWER volume requirement
                'volume_ratio': volume_ratio > 0.4,  # MUCH LOWER volume ratio requirement
                
                # Volatility conditions - OPTIMIZED FOR BETTER RETURNS
                'volatility': volatility > 0.05 and volatility < 1.2,  # MUCH MORE relaxed volatility range
                'volatility_20': volatility_20 > 0.05 and volatility_20 < 1.5,  # MUCH MORE relaxed short-term volatility
                
                # Bollinger Bands - OPTIMIZED FOR BETTER RETURNS
                'bollinger': bb_position < 0.95,  # MUCH LESS conservative Bollinger
                'bb_width': bb_width > 0.01,  # MUCH LOWER BB width requirement
                
                # Market regime - OPTIMIZED FOR BETTER RETURNS
                'market_regime': market_trend in ['bullish', 'neutral', 'bearish'],  # Accept all market regimes
                
                # Advanced features - OPTIMIZED FOR BETTER RETURNS
                'advanced_score': advanced_score > -0.5,  # MUCH MORE relaxed advanced score requirement
                
                # New advanced conditions for maximum performance
                'volume_price_confirmation': volume_trend > 0.3 and momentum_5 > 0.001,  # MUCH MORE relaxed volume-price confirmation
            }
            
            # ===== OPTIMIZED WEIGHTS FOR MAXIMUM PERFORMANCE =====
            base_weights = {
                'prediction': 2.0,  # Higher weight for prediction
                'ensemble_confidence': 2.5,  # Higher weight for confidence
                'model_reliability': 2.2,  # Higher weight for reliability
                'rsi': 0.8,  # Higher weight for RSI
                'macd': 0.8,
                'trend': 1.0,
                'momentum_5': 0.6,
                'momentum_10': 0.7,
                'momentum_20': 0.4,
                'volume_trend': 0.5,
                'volume_ratio': 0.4,
                'volatility': 0.3,
                'volatility_20': 0.3,
                'bollinger': 0.2,
                'bb_width': 0.2,
                'market_regime': 0.3,
                'advanced_score': 1.0,
                'volume_price_confirmation': 1.2,  # New - moderate
            }
            
            # Adjust weights based on market conditions
            if market_trend == 'bullish':
                base_weights['trend'] *= 1.5
                base_weights['momentum_5'] *= 1.3
                base_weights['momentum_10'] *= 1.3
            elif market_trend == 'bearish':
                base_weights['rsi'] *= 1.5  # Oversold conditions more important
                base_weights['bollinger'] *= 1.3
            
            # Calculate weighted score
            total_score = sum(base_weights[cond] for cond, met in conditions.items() if met)
            max_score = sum(base_weights.values())
            score_ratio = total_score / max_score
            
            # ===== OPTIMIZED THRESHOLDS =====
            # Use optimized parameters if available
            if optimized_params:
                required_score = optimized_params.get('required_score', 0.15)
                min_conditions = optimized_params.get('min_conditions', 2)
            else:
                # Optimized thresholds for maximum performance - MUCH MORE RELAXED
                required_score = 0.15   # MUCH LOWER score requirement - 15% of maximum
                min_conditions = 2      # Need at least 2 conditions to enter (MUCH EASIER)
            
            # Log conditions for debugging
            logger.info(f"[ADVANCED] {symbol} Entry Analysis:")
            logger.info(f"  Prediction: {prediction:.4f}, Confidence: {ensemble_confidence:.3f}, Reliability: {model_reliability:.3f}")
            logger.info(f"  RSI: {rsi:.1f}, MACD: {macd_value:.4f}, Trend: {current_price:.2f} vs {sma_20:.2f}")
            logger.info(f"  Momentum: 5d={momentum_5:.3f}, 10d={momentum_10:.3f}, 20d={momentum_20:.3f}")
            logger.info(f"  Volume: {volume_trend:.2f}, Volatility: {volatility:.2f}, BB Position: {bb_position:.2f}")
            logger.info(f"  Market: {market_trend}, Advanced Score: {advanced_score:.2f}")
            logger.info(f"  Conditions Met: {sum(conditions.values())}/{len(conditions)}, Score Ratio: {score_ratio:.2f}")
            
            return score_ratio >= required_score and sum(conditions.values()) >= min_conditions
            
        except Exception as e:
            logger.error(f"Error checking entry conditions for {symbol}: {str(e)}")
            return False

    def check_exit_conditions(self, symbol, position, current_price, current_date=None):
        """
        Check the enhanced exit conditions with trailing stops and dynamic profit taking.

        Args:
            symbol (str): The stock symbol.
            position (Dict[str, Any]): The current position.
            current_price (float): The current price.
            current_date (str): The current date for backtest (optional).

        Returns:
            bool: True if the exit conditions are met, False otherwise.
        """
        try:
            # Get position data
            entry_price = position['entry_price']
            entry_date = pd.to_datetime(position['entry_date'])
            
            # Use provided date or current date
            if current_date:
                current_date = pd.to_datetime(current_date)
            else:
                current_date = pd.to_datetime('today')
                
            holding_period = (current_date - entry_date).days
            
            # Calculate profit/loss
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Get recent data for technical analysis
            data = self.data_manager.load_data(symbol)
            if data is None:
                return False
            
            # Calculate technical indicators
            rsi = self.calculate_rsi(data['Close']).iloc[-1]
            macd, signal = self.calculate_macd(data['Close'])
            macd_value = macd.iloc[-1]
            signal_value = signal.iloc[-1]
            
            # Calculate moving averages
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            
            # Calculate volatility
            returns = data['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)
            
            # Trailing stop logic - less aggressive for better performance
            trailing_stop_pct = self.config.get('trailing_stop_pct', 0.05)  # Increased from 0.02
            if 'peak_price' not in position:
                position['peak_price'] = entry_price
            
            # Update peak price if current price is higher
            if current_price > position['peak_price']:
                position['peak_price'] = current_price
            
            # Calculate trailing stop
            trailing_stop_price = position['peak_price'] * (1 - trailing_stop_pct)
            
            # Optimized exit conditions for maximum performance - LESS AGGRESSIVE
            exit_conditions = {
                'stop_loss': pnl_pct <= -self.config.get('stop_loss_pct', 0.04),  # LESS AGGRESSIVE stop loss
                'take_profit': pnl_pct >= self.config.get('take_profit_pct', 0.12),  # HIGHER take profit for better returns
                'trailing_stop': current_price <= trailing_stop_price and pnl_pct > 0.04,  # LESS AGGRESSIVE trailing stop
                'max_holding': holding_period >= self.config.get('max_holding_period', 45),  # LONGER holding period
                'overbought': rsi > 78,  # LESS AGGRESSIVE overbought condition
                'macd_bearish': macd_value < signal_value and macd_value < -0.3,  # LESS AGGRESSIVE MACD condition
                'trend_reversal': current_price < sma_20 and sma_20 < sma_50 and pnl_pct > 0.025,  # LESS AGGRESSIVE trend reversal
                'high_volatility': volatility > 0.9 and pnl_pct > 0.06,  # LESS AGGRESSIVE volatility condition
                'momentum_loss': pnl_pct > 0.06 and (current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5] < -0.025,  # LESS AGGRESSIVE momentum loss
                'volume_spike': data['Volume'].iloc[-1] > data['Volume'].rolling(window=20).mean().iloc[-1] * 3.0 and pnl_pct > 0.05  # LESS AGGRESSIVE volume spike exit
            }
            
            # Optimized weights for maximum performance
            weights = {
                'stop_loss': 12.0,      # Critical - increased
                'take_profit': 10.0,    # Very important - increased
                'trailing_stop': 7.0,   # Important - increased
                'overbought': 6.0,      # Important - increased
                'macd_bearish': 5.0,    # Moderate - increased
                'trend_reversal': 4.0,  # Moderate - increased
                'max_holding': 3.0,     # Less important - increased
                'high_volatility': 2.0, # Less important - increased
                'momentum_loss': 2.0,   # Less important - increased
                'volume_spike': 3.0     # New - moderate
            }
            
            # Calculate weighted score
            total_score = sum(weights[cond] for cond, met in exit_conditions.items() if met)
            max_score = sum(weights.values())
            exit_score_ratio = total_score / max_score
            
            # Log exit conditions for debugging (reduced for speed)
            logger.debug(f"[DEBUG] {symbol} Exit: PnL={pnl_pct:.2%}, RSI={rsi:.1f}, Score={exit_score_ratio:.2f}")
            
            # Exit if any critical condition is met or score ratio is high enough
            critical_conditions = ['stop_loss', 'take_profit']  # Removed trailing_stop from critical
            critical_exit = any(exit_conditions[cond] for cond in critical_conditions)
            
            return critical_exit or exit_score_ratio >= 0.45  # LESS AGGRESSIVE exits for better performance
            
        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol}: {str(e)}")
            return False

    def ensemble_predict(self, symbol):
        """
        Get an ensemble prediction from the LSTM, XGBoost, and NN models.

        This method trains the models if they are not already fitted and then
        combines their predictions using dynamic weighting.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Tuple[Optional[float], Dict[str, float]]: A tuple containing the ensemble
            prediction and a dictionary of individual model predictions.
        """
        predictions = {}
        
        # Clean up any corrupted model files first
        self._cleanup_corrupted_models(symbol)
        
        # Prepare data with consistent feature engineering
        data = self.data_manager.load_data(symbol)
        if data is None or len(data) < self.config["sequence_length"]:
            self.logger.error(f"Insufficient data for ensemble prediction for {symbol}")
            return None, predictions
        
        # Use the same feature engineering as during training
        if hasattr(self, 'feature_engineering') and self.config.get("enable_advanced_features", True) and symbol in self.selected_features and self.selected_features[symbol]:
            # Use advanced feature engineering with selected features
            data = self.feature_engineering.create_all_features(data)
            selected_features = self.selected_features[symbol]
            feature_data = data[selected_features]
        else:
            # Use basic feature engineering (consistent with training)
            data['Returns'] = data['Close'].pct_change()
            data['Volume_Change'] = data['Volume'].pct_change()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])
            data = data.dropna()
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Volume_Change', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal']
            feature_data = data[features]
        
        # Ensure we have enough data
        if len(feature_data) < self.config["sequence_length"]:
            self.logger.error(f"Insufficient feature data for {symbol}")
            return None, predictions
        
        # Scale the data consistently
        if symbol not in self.scalers:
            self.scalers[symbol] = MinMaxScaler()
            self.scalers[symbol].fit(feature_data.values)
        
        scaled_data = self.scalers[symbol].transform(feature_data.values)
        latest_sequence = scaled_data[-self.config["sequence_length"]:]
        latest_sequence = latest_sequence.reshape(1, self.config["sequence_length"], -1)
        
        # LSTM prediction
        try:
            if symbol not in self.models or not isinstance(self.models[symbol], LSTMModel):
                self.models[symbol] = LSTMModel(
                    input_size=latest_sequence.shape[2],
                    hidden_size=self.config["hidden_size"]
                )
                model_path = os.path.join(self.config["models_dir"], f"{symbol}_lstm.pth")
                if os.path.exists(model_path):
                    try:
                        self.models[symbol].load_weights(model_path)
                    except Exception as e:
                        self.logger.warning(f"Could not load LSTM weights for {symbol}: {e}")
                else:
                    # Train if not found
                    self.logger.info(f"No LSTM model found for {symbol}, training...")
                    X_train, X_test, y_train, y_test = self.prepare_data(symbol)
                    self.models[symbol].train_model(
                        X_train, y_train,
                        epochs=self.config["training_epochs"],
                        batch_size=self.config["batch_size"],
                        learning_rate=self.config["learning_rate"]
                    )
                    self.models[symbol].save_weights(model_path)
            
            self.models[symbol].eval()
            with torch.no_grad():
                lstm_pred = self.models[symbol](torch.FloatTensor(latest_sequence)).numpy()[0][0]
            predictions['lstm'] = lstm_pred
        except Exception as e:
            self.logger.warning(f"LSTM prediction failed for {symbol}: {e}")
        
        # NN prediction with consistent feature count
        try:
            if symbol+"_nn" not in self.models:
                input_size = latest_sequence.shape[1] * latest_sequence.shape[2]  # sequence_length * num_features
                nn_model = CustomNeuralNetwork(
                    input_size=input_size,
                    hidden_size=self.config["hidden_size"],
                    output_size=1,
                    learning_rate=self.config["learning_rate"]
                )
                model_path = os.path.join(self.config["models_dir"], f"{symbol}_nn.pkl")
                if os.path.exists(model_path):
                    try:
                        nn_model.load_weights(model_path)
                        # Verify weight compatibility
                        if nn_model.weights_input_hidden.shape[0] != input_size:
                            self.logger.warning(f"NN weights shape mismatch for {symbol}: expected {input_size}, got {nn_model.weights_input_hidden.shape[0]}. Retraining.")
                            nn_model = CustomNeuralNetwork(
                                input_size=input_size,
                                hidden_size=self.config["hidden_size"],
                                output_size=1,
                                learning_rate=self.config["learning_rate"]
                            )
                            # Retrain with correct dimensions
                            X_train, X_test, y_train, y_test = self.prepare_data(symbol)
                            if isinstance(X_train, torch.Tensor):
                                X_train = X_train.detach().cpu().numpy()
                            if isinstance(y_train, torch.Tensor):
                                y_train = y_train.detach().cpu().numpy()
                            X_train = X_train.reshape(X_train.shape[0], -1)
                            nn_model.train(X_train, y_train,
                                epochs=self.config["training_epochs"],
                                batch_size=self.config["batch_size"])
                            nn_model.save_weights(model_path)
                    except Exception as e:
                        self.logger.warning(f"Could not load NN weights for {symbol}: {e}")
                        # Train if loading fails
                        X_train, X_test, y_train, y_test = self.prepare_data(symbol)
                        if isinstance(X_train, torch.Tensor):
                            X_train = X_train.detach().cpu().numpy()
                        if isinstance(y_train, torch.Tensor):
                            y_train = y_train.detach().cpu().numpy()
                        X_train = X_train.reshape(X_train.shape[0], -1)
                        nn_model.train(X_train, y_train,
                            epochs=self.config["training_epochs"],
                            batch_size=self.config["batch_size"])
                        nn_model.save_weights(model_path)
                else:
                    # Train if not found
                    self.logger.info(f"No NN model found for {symbol}, training...")
                    X_train, X_test, y_train, y_test = self.prepare_data(symbol)
                    if isinstance(X_train, torch.Tensor):
                        X_train = X_train.detach().cpu().numpy()
                    if isinstance(y_train, torch.Tensor):
                        y_train = y_train.detach().cpu().numpy()
                    X_train = X_train.reshape(X_train.shape[0], -1)
                    nn_model.train(X_train, y_train,
                        epochs=self.config["training_epochs"],
                        batch_size=self.config["batch_size"])
                    nn_model.save_weights(model_path)
                
                self.models[symbol+"_nn"] = nn_model
            
            # Prepare input with correct shape
            nn_input = latest_sequence.reshape(1, -1)
            if isinstance(nn_input, torch.Tensor):
                nn_input = nn_input.detach().cpu().numpy()
            if not isinstance(nn_input, np.ndarray):
                nn_input = np.array(nn_input, dtype=np.float32)
            else:
                nn_input = nn_input.astype(np.float32)
            
            # Ensure correct shape
            expected_input_size = self.models[symbol+"_nn"].weights_input_hidden.shape[0]
            if nn_input.shape[1] != expected_input_size:
                self.logger.warning(f"NN input shape mismatch: got {nn_input.shape[1]}, expected {expected_input_size}")
                return None, predictions
            
            nn_pred = self.models[symbol+"_nn"].predict(nn_input)[0][0]
            predictions['nn'] = nn_pred
        except Exception as e:
            self.logger.warning(f"NN prediction failed for {symbol}: {e}")
        
        # XGBoost prediction with consistent feature count
        try:
            if symbol+"_xgb" not in self.models:
                # XGBoost doesn't use input_size, hidden_size, output_size
                xgb_model = XGBoostModel(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                model_path = os.path.join(self.config["models_dir"], f"{symbol}_xgb.pkl")
                if os.path.exists(model_path):
                    try:
                        xgb_model.load_weights(model_path)
                    except Exception as e:
                        self.logger.warning(f"Could not load XGBoost weights for {symbol}: {e}")
                        # Train if loading fails
                        X_train, X_test, y_train, y_test = self.prepare_data(symbol)
                        if isinstance(X_train, torch.Tensor):
                            X_train = X_train.detach().cpu().numpy()
                        if isinstance(y_train, torch.Tensor):
                            y_train = y_train.detach().cpu().numpy()
                        X_train = X_train.reshape(X_train.shape[0], -1)
                        xgb_model.fit(X_train, y_train)
                        xgb_model.save(model_path)
                else:
                    # Train if not found
                    self.logger.info(f"No XGBoost model found for {symbol}, training...")
                    X_train, X_test, y_train, y_test = self.prepare_data(symbol)
                    if isinstance(X_train, torch.Tensor):
                        X_train = X_train.detach().cpu().numpy()
                    if isinstance(y_train, torch.Tensor):
                        y_train = y_train.detach().cpu().numpy()
                    X_train = X_train.reshape(X_train.shape[0], -1)
                    xgb_model.fit(X_train, y_train)
                    xgb_model.save(model_path)
                
                self.models[symbol+"_xgb"] = xgb_model
            
            # Prepare input with correct shape
            xgb_input = latest_sequence.reshape(1, -1)
            if isinstance(xgb_input, torch.Tensor):
                xgb_input = xgb_input.detach().cpu().numpy()
            
            # Ensure input is properly formatted
            if not isinstance(xgb_input, np.ndarray):
                xgb_input = np.array(xgb_input, dtype=np.float32)
            else:
                xgb_input = xgb_input.astype(np.float32)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(xgb_input)) or np.any(np.isinf(xgb_input)):
                self.logger.warning(f"XGBoost input contains NaN or infinite values for {symbol}")
                raise ValueError("Invalid input data")
            
            xgb_pred = self.models[symbol+"_xgb"].predict(xgb_input)
            if isinstance(xgb_pred, np.ndarray) and len(xgb_pred) > 0:
                predictions['xgb'] = xgb_pred[0]
            else:
                predictions['xgb'] = xgb_pred
            
        except Exception as e:
            self.logger.warning(f"XGBoost prediction failed for {symbol}: {e}")
            # Don't add XGBoost prediction to the ensemble if it fails
        
        # Dynamic ensemble weighting with optimized weights
        if predictions:
            try:
                # Load optimized ensemble weights if available
                optimized_params = self._load_optimized_parameters(symbol)
                ensemble_params = optimized_params.get("ensemble", {})
                
                # Use optimized weights if available, otherwise use dynamic ensemble
                if ensemble_params and 'weights' in ensemble_params:
                    weights = ensemble_params['weights']
                    self.logger.info(f"Using optimized ensemble weights for {symbol}: {weights}")
                    
                    # Calculate weighted ensemble prediction with confidence weighting
                    ensemble_pred = 0.0
                    total_weight = 0.0
                    confidence_scores = {}
                    
                    # Calculate confidence scores for each model
                    for model_name, pred in predictions.items():
                        if model_name in weights:
                            # Calculate confidence based on prediction consistency
                            confidence = min(1.0, max(0.1, weights[model_name]))
                            confidence_scores[model_name] = confidence
                            ensemble_pred += pred * confidence
                            total_weight += confidence
                    
                    if total_weight > 0:
                        ensemble_pred /= total_weight
                    else:
                        # Fallback to simple average if no valid weights
                        ensemble_pred = sum(predictions.values()) / len(predictions)
                    
                    self.logger.info(f"[ENSEMBLE] {data.index[-1]} - {symbol} Prediction: {ensemble_pred:.4f}, Current: {data['Close'].iloc[-1]:.4f}")
                    for model_name, pred in predictions.items():
                        self.logger.info(f"  {model_name.upper()}: {pred:.4f}")
                    
                    return ensemble_pred, predictions
                    
                elif hasattr(self, 'dynamic_ensemble'):
                    # Get current price for performance tracking
                    current_price = data['Close'].iloc[-1]
                    
                    # Get market conditions for dynamic weighting
                    market_conditions = self._get_market_conditions(data)
                    
                    # Update ensemble weights based on recent performance
                    updated_weights = self.dynamic_ensemble.update_weights(
                        predictions, current_price, market_conditions
                    )
                    
                    # Get weighted ensemble prediction
                    ensemble_pred = self.dynamic_ensemble.get_ensemble_prediction(predictions)
                    
                    # Get ensemble confidence
                    confidence = self.dynamic_ensemble.get_ensemble_confidence(predictions)
                    
                    self.logger.info(f"[ENSEMBLE] {data.index[-1]} - {symbol} Prediction: {ensemble_pred:.4f}, Current: {current_price:.4f}")
                    self.logger.info(f"  LSTM: {predictions.get('lstm', 0):.4f}")
                    self.logger.info(f"  NN: {predictions.get('nn', 0):.4f}")
                    self.logger.info(f"  XGB: {predictions.get('xgb', 0):.4f}")
                    
                    return ensemble_pred, predictions
                else:
                    # Enhanced fallback with weighted average based on model reliability
                    self.logger.warning(f"Dynamic ensemble failed for {symbol}, using enhanced fallback: {list(predictions.keys())}")
                    
                    # Calculate weighted average based on model type reliability
                    model_weights = {
                        'lstm': 0.4,    # LSTM typically performs well for time series
                        'xgb': 0.35,    # XGBoost is reliable for structured data
                        'nn': 0.25      # Neural network as backup
                    }
                    
                    ensemble_pred = 0.0
                    total_weight = 0.0
                    
                    for model_name, pred in predictions.items():
                        weight = model_weights.get(model_name, 0.2)  # Default weight for unknown models
                        ensemble_pred += pred * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        ensemble_pred /= total_weight
                    else:
                        ensemble_pred = sum(predictions.values()) / len(predictions)
                    
                    self.logger.info(f"[ENSEMBLE] {data.index[-1]} - {symbol} Prediction: {ensemble_pred:.4f}, Current: {data['Close'].iloc[-1]:.4f}")
                    for model_name, pred in predictions.items():
                        self.logger.info(f"  {model_name.upper()}: {pred:.4f}")
                    
                    return ensemble_pred, predictions
                    
            except Exception as e:
                self.logger.warning(f"Dynamic ensemble failed for {symbol}, falling back to simple average: {e}")
                ensemble_pred = sum(predictions.values()) / len(predictions)
                
                self.logger.info(f"[ENSEMBLE] {data.index[-1]} - {symbol} Prediction: {ensemble_pred:.4f}, Current: {data['Close'].iloc[-1]:.4f}")
                for model_name, pred in predictions.items():
                    self.logger.info(f"  {model_name.upper()}: {pred:.4f}")
                
                return ensemble_pred, predictions
        else:
            self.logger.error(f"No predictions available for {symbol}")
            return None, predictions
    
    def _get_market_conditions(self, data):
        """
        Extract market conditions for dynamic ensemble weighting.

        Args:
            data (pd.DataFrame): The market data.

        Returns:
            Dict[str, float]: A dictionary of market conditions.
        """
        try:
            # Calculate volatility
            returns = data['Close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Calculate trend strength
            sma_20 = data['Close'].rolling(20).mean()
            sma_50 = data['Close'].rolling(50).mean()
            trend_strength = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
            
            # Calculate volume ratio
            volume_sma = data['Volume'].rolling(20).mean()
            volume_ratio = data['Volume'].iloc[-1] / volume_sma.iloc[-1]
            
            return {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'volume_ratio': volume_ratio
            }
        except Exception as e:
            logger.warning(f"Error calculating market conditions: {e}")
            return {}
    
    def run_hyperparameter_optimization(self, symbol: str):
        """
        Run hyperparameter optimization for all models.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of the best hyperparameters, or None if an error occurs.
        """
        if not hasattr(self, 'hyperparameter_optimizer'):
            logger.error("Hyperparameter optimizer not initialized")
            return None
        
        try:
            logger.info(f"Starting hyperparameter optimization for {symbol}")
            best_params = self.hyperparameter_optimizer.optimize_all_models(symbol)
            
            # Save optimized parameters
            params_file = f"hyperparameter_results/{symbol}_optimized_params.json"
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(best_params, f, indent=4)
            
            # Save selected features after optimization
            self._save_selected_features(symbol)
            
            logger.info(f"Hyperparameter optimization completed for {symbol}")
            return best_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization for {symbol}: {e}")
            return None
    
    def run_walk_forward_backtest(self, symbol: str, start_date: str, end_date: str):
        """
        Run walk-forward backtesting for a symbol.

        Args:
            symbol (str): The stock symbol.
            start_date (str): The start date for the backtest.
            end_date (str): The end date for the backtest.

        Returns:
            Optional[Dict[str, Any]]: The results of the walk-forward backtest, or None if an error occurs.
        """
        if not hasattr(self, 'walk_forward_backtest'):
            logger.error("Walk-forward backtest not initialized")
            return None
        
        try:
            logger.info(f"Starting walk-forward backtest for {symbol}")
            results = self.walk_forward_backtest.run_walk_forward_backtest(
                symbol, start_date, end_date
            )
            
            logger.info(f"Walk-forward backtest completed for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error in walk-forward backtest for {symbol}: {e}")
            return None
    
    def generate_performance_report(self, symbol: str):
        """
        Generate a comprehensive performance report.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Optional[str]: The performance report, or None if an error occurs.
        """
        if not hasattr(self, 'model_monitor'):
            logger.error("Model monitor not initialized")
            return None
        
        try:
            # Calculate performance metrics for all models
            for model_name in ['lstm', 'nn', 'xgb', 'ensemble']:
                self.model_monitor.calculate_performance_metrics(model_name)
            
            # Generate report
            report = self.model_monitor.generate_performance_report('ensemble')
            
            # Add ensemble summary
            if hasattr(self, 'dynamic_ensemble'):
                ensemble_summary = self.dynamic_ensemble.get_weight_summary()
                report += f"\n{ensemble_summary}"
            
            # Save report
            report_file = f"monitoring_data/{symbol}_performance_report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Performance report generated for {symbol}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report for {symbol}: {e}")
            return None
    
    def get_advanced_metrics(self, symbol: str):
        """
        Get advanced risk and performance metrics.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Optional[Tuple[Dict[str, Any], str]]: A tuple containing the metrics and the risk report,
            or None if an error occurs.
        """
        if not hasattr(self, 'risk_metrics'):
            logger.error("Risk metrics not initialized")
            return None
        
        try:
            # Load data
            data = self.data_manager.load_data(symbol)
            if data is None:
                return None
            
            # Calculate returns
            returns = self.risk_metrics.calculate_returns(data['Close'])
            
            # Calculate all metrics
            metrics = self.risk_metrics.calculate_all_metrics(returns)
            
            # Generate risk report
            risk_report = self.risk_metrics.generate_risk_report(returns, f"{symbol} Strategy")
            
            # Save metrics
            metrics_file = f"monitoring_data/{symbol}_risk_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Advanced metrics calculated for {symbol}")
            return metrics, risk_report
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics for {symbol}: {e}")
            return None

    def _save_selected_features(self, symbol):
        """
        Save the selected features for a symbol to disk.

        Args:
            symbol (str): The stock symbol.
        """
        try:
            features = self.selected_features.get(symbol)
            if features:
                models_dir = self.config.get("models_dir", "models")
                os.makedirs(models_dir, exist_ok=True)
                path = os.path.join(models_dir, f"{symbol}_selected_features.json")
                with open(path, "w", encoding='utf-8') as f:
                    json.dump(features, f)
                logger.info(f"Saved selected features for {symbol} to {path}")
        except Exception as e:
            logger.error(f"Failed to save selected features for {symbol}: {e}")

    def _load_selected_features(self, symbol):
        """
        Load the selected features for a symbol from disk.

        Args:
            symbol (str): The stock symbol.
        """
        try:
            models_dir = self.config.get("models_dir", "models")
            path = os.path.join(models_dir, f"{symbol}_selected_features.json")
            if os.path.exists(path):
                with open(path, "r", encoding='utf-8') as f:
                    self.selected_features[symbol] = json.load(f)
                logger.info(f"Loaded selected features for {symbol} from {path}")
        except Exception as e:
            logger.error(f"Failed to load selected features for {symbol}: {e}")

    def _load_selected_features_all_symbols(self):
        """Load the selected features for all symbols in the config."""
        for symbol in self.config.get("symbols", []):
            self._load_selected_features(symbol)

    def _save_models_and_scalers(self, symbol):
        """
        Save all models and scalers for a symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            models_dir = self.config.get("models_dir", "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Save LSTM model
            if symbol in self.models and isinstance(self.models[symbol], LSTMModel):
                lstm_path = os.path.join(models_dir, f"{symbol}_lstm.pth")
                self.models[symbol].save_weights(lstm_path)
                logger.info(f"LSTM model saved successfully to {lstm_path}")
            
            # Save Neural Network model
            if symbol+"_nn" in self.models:
                nn_path = os.path.join(models_dir, f"{symbol}_nn.pkl")
                self.models[symbol+"_nn"].save_weights(nn_path)
                logger.info(f"Neural Network model saved successfully to {nn_path}")
            
            # Save XGBoost model
            if symbol+"_xgb" in self.models:
                xgb_path = os.path.join(models_dir, f"{symbol}_xgb.json")
                self.models[symbol+"_xgb"].save_weights(xgb_path)
                logger.info(f"XGBoost model saved successfully to {xgb_path}")
            
            # Save scaler
            if symbol in self.scalers:
                scaler_path = os.path.join(models_dir, f"{symbol}_scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[symbol], f)
                logger.info(f"Scaler saved successfully to {scaler_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving models and scalers for {symbol}: {e}")
            return False

    def _cleanup_corrupted_models(self, symbol):
        """
        Clean up corrupted model files for a symbol.

        Args:
            symbol (str): The stock symbol.
        """
        try:
            models_dir = self.config.get("models_dir", "models")
            
            # List of model files to check
            model_files = [
                os.path.join(models_dir, f"{symbol}_lstm.pth"),
                os.path.join(models_dir, f"{symbol}_nn.pkl"),
                os.path.join(models_dir, f"{symbol}_xgb.json"),
                os.path.join(models_dir, f"{symbol}_xgb.json_metadata.json")
            ]
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    try:
                        # Try to open and read the file to check if it's corrupted
                        if model_file.endswith('.json'):
                            with open(model_file, 'r', encoding='utf-8') as f:
                                json.load(f)  # Test if JSON is valid
                        elif model_file.endswith('.pth'):
                            torch.load(model_file)  # Test if PyTorch file is valid
                        elif model_file.endswith('.pkl'):
                            with open(model_file, 'rb') as f:
                                pickle.load(f)  # Test if pickle file is valid
                    except Exception as e:
                        self.logger.warning(f"Corrupted model file detected: {model_file}, removing: {e}")
                        try:
                            os.remove(model_file)
                        except Exception as remove_error:
                            self.logger.error(f"Could not remove corrupted file {model_file}: {remove_error}")
                            
        except Exception as e:
            self.logger.error(f"Error cleaning up corrupted models for {symbol}: {e}")

def main():
    """
    The main function to run the advanced trading bot.

    This function parses command-line arguments and runs the appropriate
    action, such as backtesting, optimization, or live trading.
    """
    parser = argparse.ArgumentParser(description="Advanced Trading Bot with Ensemble Learning")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--backtest", action="store_true", help="Run backtest instead of live trading")
    parser.add_argument("--start-date", type=str, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward backtesting")
    parser.add_argument("--symbol", type=str, help="Symbol to analyze")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    parser.add_argument("--metrics", action="store_true", help="Calculate advanced risk metrics")
    
    args = parser.parse_args()
    
    # Initialize trading bot
    bot = TradingBot(config_file=args.config)
    
    # Run hyperparameter optimization
    if args.optimize and args.symbol:
        logger.info(f"Running hyperparameter optimization for {args.symbol}")
        best_params = bot.run_hyperparameter_optimization(args.symbol)
        if best_params:
            logger.info("Hyperparameter optimization completed successfully")
        return
    
    # Run walk-forward backtesting
    if args.walk_forward and args.symbol:
        start_date = args.start_date or "2023-01-01"
        end_date = args.end_date or "2024-01-01"
        logger.info(f"Running walk-forward backtest for {args.symbol}")
        results = bot.run_walk_forward_backtest(args.symbol, start_date, end_date)
        if results:
            logger.info("Walk-forward backtesting completed successfully")
        return
    
    # Generate performance report
    if args.report and args.symbol:
        logger.info(f"Generating performance report for {args.symbol}")
        report = bot.generate_performance_report(args.symbol)
        if report:
            logger.info("Performance report generated successfully")
        return
    
    # Calculate advanced metrics
    if args.metrics and args.symbol:
        logger.info(f"Calculating advanced metrics for {args.symbol}")
        metrics, report = bot.get_advanced_metrics(args.symbol)
        if metrics:
            logger.info("Advanced metrics calculated successfully")
        return
    
    # Train model for specific symbol
    if args.symbol and not any([args.optimize, args.walk_forward, args.report, args.metrics, args.backtest]):
        logger.info(f"Training model for {args.symbol}")
        success = bot.train_model(args.symbol)
        if success:
            logger.info(f"Model training completed successfully for {args.symbol}")
        else:
            logger.error(f"Model training failed for {args.symbol}")
        return
    
    # Run regular backtest or live trading
    if args.backtest:
        # Run backtest
        bot.backtest(args.start_date, args.end_date)
    else:
        # Run live trading
        bot.run()

if __name__ == "__main__":
    main()

        
