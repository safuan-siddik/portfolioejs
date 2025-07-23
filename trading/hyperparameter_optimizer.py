"""
Hyperparameter Optimization Module
==================================
This module provides advanced hyperparameter optimization using Optuna for all
model types in the ensemble.

Classes:
- HyperparameterOptimizer: The main class for running optimizations.
"""

import optuna
import numpy as np
import pandas as pd
import logging
import pickle
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F

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
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
    
    def load(self, filepath):
        """Load the model."""
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading XGBoost model: {e}")
            self.model = None

logger = logging.getLogger("HyperparameterOptimizer")

class HyperparameterOptimizer:
    """
    An advanced hyperparameter optimizer using Optuna for the ensemble models.

    This class is responsible for finding the best hyperparameters for the LSTM,
    Neural Network, and XGBoost models, as well as the ensemble itself.
    """
    
    def __init__(self, data_manager, config: Dict[str, Any]):
        """
        Initialize the hyperparameter optimizer.

        Args:
            data_manager: An instance of the MarketDataManager.
            config (Dict[str, Any]): The configuration dictionary.
        """
        self.data_manager = data_manager
        self.config = config
        self.best_params = {}
        
    def optimize_lstm_hyperparameters(self, symbol: str, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize the LSTM hyperparameters using Optuna.

        Args:
            symbol (str): The stock symbol to optimize for.
            n_trials (int): The number of optimization trials to run.

        Returns:
            Dict[str, Any]: A dictionary of the best hyperparameters.
        """
        
        def objective(trial):
            # Define hyperparameter search space
            hidden_size = trial.suggest_int('hidden_size', 16, 128, step=16)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
            sequence_length = trial.suggest_int('sequence_length', 10, 50, step=5)
            
            try:
                # Prepare data
                X_train, X_test, y_train, y_test = self._prepare_data_for_optimization(symbol, sequence_length)
                
                # Initialize model
                model = LSTMModel(
                    input_size=X_train.shape[2],
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout
                )
                
                # Prepare DataLoaders
                train_loader, val_loader = self._create_dataloaders(X_train, y_train, batch_size)
                
                # Train model
                history = train_model(
                    model, train_loader, val_loader,
                    epochs=50,  # Reduced for optimization
                    learning_rate=learning_rate
                )
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test)
                    mse = F.mse_loss(y_pred, y_test).item()
                
                return mse
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        self.best_params['lstm'] = study.best_params
        logger.info(f"Best LSTM params: {study.best_params}")
        logger.info(f"Best LSTM score: {study.best_value}")
        
        return study.best_params
    
    def optimize_nn_hyperparameters(self, symbol: str, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize the Neural Network hyperparameters using Optuna.

        Args:
            symbol (str): The stock symbol to optimize for.
            n_trials (int): The number of optimization trials to run.

        Returns:
            Dict[str, Any]: A dictionary of the best hyperparameters.
        """
        
        def objective(trial):
            # Define hyperparameter search space
            hidden_size = trial.suggest_int('hidden_size', 16, 128, step=16)
            learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
            sequence_length = trial.suggest_int('sequence_length', 10, 50, step=5)
            
            try:
                # Prepare data
                X_train, X_test, y_train, y_test = self._prepare_data_for_optimization(symbol, sequence_length)
                
                # Reshape for NN
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)
                
                # Initialize model
                model = CustomNeuralNetwork(
                    input_size=X_train.shape[1],
                    hidden_size=hidden_size,
                    output_size=1,
                    learning_rate=learning_rate
                )
                
                # Train model
                history = model.train(
                    X_train, y_train,
                    epochs=50,  # Reduced for optimization
                    batch_size=batch_size
                )
                
                # Evaluate
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                
                return mse
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        self.best_params['nn'] = study.best_params
        logger.info(f"Best NN params: {study.best_params}")
        logger.info(f"Best NN score: {study.best_value}")
        
        return study.best_params
    
    def optimize_xgboost_hyperparameters(self, symbol: str, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize the XGBoost hyperparameters using Optuna.

        Args:
            symbol (str): The stock symbol to optimize for.
            n_trials (int): The number of optimization trials to run.

        Returns:
            Dict[str, Any]: A dictionary of the best hyperparameters.
        """
        
        def objective(trial):
            # Define hyperparameter search space
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            subsample = trial.suggest_float('subsample', 0.6, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            min_child_weight = trial.suggest_int('min_child_weight', 1, 7)
            sequence_length = trial.suggest_int('sequence_length', 10, 50, step=5)
            
            try:
                # Prepare data
                X_train, X_test, y_train, y_test = self._prepare_data_for_optimization(symbol, sequence_length)
                
                # Reshape for XGBoost
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)
                
                # Initialize model
                model = XGBoostModel(
                    input_size=X_train.shape[1],
                    hidden_size=32,  # Not used in XGBoost
                    output_size=1,
                    learning_rate=learning_rate
                )
                
                # Update XGBoost parameters
                model.model.set_params(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight
                )
                
                # Train model
                history = model.train(
                    X_train, y_train,
                    epochs=50,  # Reduced for optimization
                    batch_size=32
                )
                
                # Evaluate
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                
                return mse
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        self.best_params['xgb'] = study.best_params
        logger.info(f"Best XGBoost params: {study.best_params}")
        logger.info(f"Best XGBoost score: {study.best_value}")
        
        return study.best_params
    
    def optimize_ensemble_hyperparameters(self, symbol: str, n_trials: int = 30) -> Dict[str, Any]:
        """
        Optimize the ensemble weights and combination strategy.

        Args:
            symbol (str): The stock symbol to optimize for.
            n_trials (int): The number of optimization trials to run.

        Returns:
            Dict[str, Any]: A dictionary of the best hyperparameters for the ensemble.
        """
        
        def objective(trial):
            # Define ensemble hyperparameter search space
            lstm_weight = trial.suggest_float('lstm_weight', 0.1, 0.8)
            nn_weight = trial.suggest_float('nn_weight', 0.1, 0.8)
            xgb_weight = trial.suggest_float('xgb_weight', 0.1, 0.8)
            
            # Ensure weights sum to 1
            total_weight = lstm_weight + nn_weight + xgb_weight
            lstm_weight /= total_weight
            nn_weight /= total_weight
            xgb_weight /= total_weight
            
            try:
                # Get predictions from optimized models
                lstm_pred, nn_pred, xgb_pred = self._get_model_predictions(symbol)
                
                # Weighted ensemble
                ensemble_pred = (lstm_weight * lstm_pred + 
                               nn_weight * nn_pred + 
                               xgb_weight * xgb_pred)
                
                # Calculate ensemble score (you can use actual vs predicted here)
                # For now, using a simple metric
                score = np.std([lstm_pred, nn_pred, xgb_pred])  # Lower diversity = better
                
                return score
                
            except Exception as e:
                logger.warning(f"Ensemble trial failed: {e}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        self.best_params['ensemble'] = study.best_params
        logger.info(f"Best ensemble params: {study.best_params}")
        logger.info(f"Best ensemble score: {study.best_value}")
        
        return study.best_params
    
    def _prepare_data_for_optimization(self, symbol: str, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare the data for hyperparameter optimization.

        Args:
            symbol (str): The stock symbol.
            sequence_length (int): The sequence length to use.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
            the training and testing data.
        """
        # This would use the same data preparation logic as the main trading bot
        # but with the specified sequence_length
        from tradingbot import TradingBot
        
        # Create a temporary bot instance for data preparation
        temp_bot = TradingBot()
        temp_bot.config["sequence_length"] = sequence_length
        
        X_train, X_test, y_train, y_test = temp_bot.prepare_data(symbol)
        return X_train, X_test, y_train, y_test
    
    def _create_dataloaders(self, X_train, y_train, batch_size: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Create DataLoaders for training and validation.

        Args:
            X_train (np.ndarray): The training data.
            y_train (np.ndarray): The training labels.
            batch_size (int): The batch size.

        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: A tuple containing
            the training and validation data loaders.
        """
        from torch.utils.data import TensorDataset, DataLoader
        
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)) # Use training data for validation
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def _get_model_predictions(self, symbol: str) -> Tuple[float, float, float]:
        """
        Get predictions from all optimized models.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Tuple[float, float, float]: A tuple containing the predictions from the LSTM,
            NN, and XGBoost models.
        """
        # This would load the optimized models and get predictions
        # For now, returning dummy values
        return 100.0, 105.0, 110.0
    
    def optimize_all_models(self, symbol: str) -> Dict[str, Any]:
        """
        Optimize the hyperparameters for all models and the ensemble.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Dict[str, Any]: A dictionary containing the best hyperparameters for all models.
        """
        logger.info(f"Starting hyperparameter optimization for {symbol}")
        
        # Optimize individual models
        lstm_params = self.optimize_lstm_hyperparameters(symbol, n_trials=30)
        nn_params = self.optimize_nn_hyperparameters(symbol, n_trials=30)
        xgb_params = self.optimize_xgboost_hyperparameters(symbol, n_trials=30)
        
        # Optimize ensemble
        ensemble_params = self.optimize_ensemble_hyperparameters(symbol, n_trials=20)
        
        # Combine all best parameters
        all_best_params = {
            'lstm': lstm_params,
            'nn': nn_params,
            'xgb': xgb_params,
            'ensemble': ensemble_params
        }
        
        logger.info(f"Hyperparameter optimization completed for {symbol}")
        return all_best_params 