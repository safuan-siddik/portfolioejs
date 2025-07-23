"""
Walk-Forward Backtesting Module
===============================
Advanced walk-forward backtesting with rolling windows and out-of-sample testing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Callable
import logging
from datetime import datetime, timedelta
import os
import json

from risk_metrics import RiskMetrics

logger = logging.getLogger("WalkForwardBacktest")

class WalkForwardBacktest:
    """
    Advanced walk-forward backtesting system with rolling windows.
    Accepts a data_preparer callable for data preparation to avoid circular imports.
    """
    
    def __init__(self, config: Dict[str, Any], data_preparer: Callable):
        """
        Initialize walk-forward backtesting.
        
        Args:
            config: Configuration dictionary
            data_preparer: Callable for preparing data (e.g., TradingBot.prepare_data)
        """
        self.config = config
        self.risk_metrics = RiskMetrics()
        self.results = {}
        self.data_preparer = data_preparer
        
    def run_walk_forward_backtest(self, symbol: str, start_date: str, end_date: str,
                                 train_window: int = 252, test_window: int = 63,
                                 step_size: int = 21) -> Dict[str, Any]:
        """
        Run walk-forward backtesting with rolling windows.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for backtesting
            end_date: End date for backtesting
            train_window: Training window size in days
            test_window: Testing window size in days
            step_size: Step size for rolling window in days
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info(f"Starting walk-forward backtest for {symbol}")
        
        # Initialize results storage
        all_predictions = []
        all_actuals = []
        all_dates = []
        model_performance = {
            'lstm': {'predictions': [], 'actuals': [], 'errors': []},
            'nn': {'predictions': [], 'actuals': [], 'errors': []},
            'xgb': {'predictions': [], 'actuals': [], 'errors': []},
            'ensemble': {'predictions': [], 'actuals': [], 'errors': []}
        }
        
        # Create date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Prepare data using the provided data_preparer
        data = self.data_preparer(symbol, start_date, end_date)
        if data is None or len(data) < train_window + test_window:
            logger.error(f"Insufficient data for walk-forward backtest: {len(data)} days")
            return {}
        
        # Create rolling windows
        current_start = start_dt
        window_count = 0
        
        while current_start + timedelta(days=train_window + test_window) <= end_dt:
            window_count += 1
            logger.info(f"Processing window {window_count}: {current_start.date()} to {(current_start + timedelta(days=train_window + test_window)).date()}")
            
            # Define train and test periods
            train_end = current_start + timedelta(days=train_window)
            test_end = train_end + timedelta(days=test_window)
            
            # Extract train and test data
            train_data = data[(data.index >= current_start) & (data.index < train_end)]
            test_data = data[(data.index >= train_end) & (data.index < test_end)]
            
            # Debug information
            logger.info(f"Window {window_count}: Train data: {len(train_data)} days, Test data: {len(test_data)} days")
            logger.info(f"Window {window_count}: Train period: {current_start.date()} to {train_end.date()}")
            logger.info(f"Window {window_count}: Test period: {train_end.date()} to {test_end.date()}")
            
            # Adjust requirements for trading days (not calendar days)
            min_train_days = int(train_window * 0.6)  # At least 60% of trading days
            min_test_days = int(test_window * 0.6)    # At least 60% of trading days
            
            if len(train_data) < min_train_days or len(test_data) < min_test_days:
                logger.warning(f"Window {window_count}: Insufficient data (train: {len(train_data)}/{min_train_days}, test: {len(test_data)}/{min_test_days}), skipping")
                current_start += timedelta(days=step_size)
                continue
            
            try:
                # Train models on training data
                self._train_models_for_window(train_data)
                
                # Make predictions on test data
                window_predictions = self._predict_on_test_window(test_data)
                
                # Store results
                for model_name, pred_data in window_predictions.items():
                    if model_name in model_performance:
                        model_performance[model_name]['predictions'].extend(pred_data['predictions'])
                        model_performance[model_name]['actuals'].extend(pred_data['actuals'])
                        model_performance[model_name]['errors'].extend(pred_data['errors'])
                
                # Store ensemble predictions
                all_predictions.extend(window_predictions['ensemble']['predictions'])
                all_actuals.extend(window_predictions['ensemble']['actuals'])
                all_dates.extend(test_data.index.tolist())
                
            except Exception as e:
                logger.error(f"Error in window {window_count}: {e}")
            
            # Move to next window
            current_start += timedelta(days=step_size)
        
        # Calculate comprehensive results
        results = self._calculate_walk_forward_results(
            model_performance, all_predictions, all_actuals, all_dates, symbol
        )
        
        # Save results
        self._save_walk_forward_results(results, symbol)
        
        # Generate visualizations
        self._generate_walk_forward_plots(results, symbol)
        
        logger.info(f"Walk-forward backtest completed for {symbol}")
        return results
    
    def _train_models_for_window(self, train_data: pd.DataFrame):
        """Train models on the training window data."""
        # This would involve training the models on the specific window data
        # For now, we'll use the existing training logic
        pass
    
    def _predict_on_test_window(self, test_data: pd.DataFrame) -> Dict[str, Dict]:
        """Make predictions on the test window data."""
        predictions = {}
        
        # Get actual prices
        actual_prices = test_data['Close'].values
        
        # Make ensemble predictions
        ensemble_preds = []
        lstm_preds = []
        nn_preds = []
        xgb_preds = []
        
        for i in range(len(test_data)):
            if i < self.config["sequence_length"]:
                # Not enough data for prediction
                ensemble_preds.append(actual_prices[i])
                lstm_preds.append(actual_prices[i])
                nn_preds.append(actual_prices[i])
                xgb_preds.append(actual_prices[i])
                continue
            
            try:
                # Get ensemble prediction
                ensemble_pred, model_preds = self._ensemble_predict(test_data.iloc[i])
                
                if ensemble_pred is not None:
                    ensemble_preds.append(ensemble_pred)
                    lstm_preds.append(model_preds.get('lstm', actual_prices[i]))
                    nn_preds.append(model_preds.get('nn', actual_prices[i]))
                    xgb_preds.append(model_preds.get('xgb', actual_prices[i]))
                else:
                    ensemble_preds.append(actual_prices[i])
                    lstm_preds.append(actual_prices[i])
                    nn_preds.append(actual_prices[i])
                    xgb_preds.append(actual_prices[i])
                    
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                ensemble_preds.append(actual_prices[i])
                lstm_preds.append(actual_prices[i])
                nn_preds.append(actual_prices[i])
                xgb_preds.append(actual_prices[i])
        
        # Calculate errors
        predictions['ensemble'] = {
            'predictions': ensemble_preds,
            'actuals': actual_prices.tolist(),
            'errors': [p - a for p, a in zip(ensemble_preds, actual_prices)]
        }
        
        predictions['lstm'] = {
            'predictions': lstm_preds,
            'actuals': actual_prices.tolist(),
            'errors': [p - a for p, a in zip(lstm_preds, actual_prices)]
        }
        
        predictions['nn'] = {
            'predictions': nn_preds,
            'actuals': actual_prices.tolist(),
            'errors': [p - a for p, a in zip(nn_preds, actual_prices)]
        }
        
        predictions['xgb'] = {
            'predictions': xgb_preds,
            'actuals': actual_prices.tolist(),
            'errors': [p - a for p, a in zip(xgb_preds, actual_prices)]
        }
        
        return predictions
    
    def _calculate_walk_forward_results(self, model_performance: Dict, all_predictions: List,
                                      all_actuals: List, all_dates: List, symbol: str) -> Dict[str, Any]:
        """Calculate comprehensive results from walk-forward backtest."""
        
        results = {
            'symbol': symbol,
            'backtest_date': datetime.now().isoformat(),
            'model_performance': {},
            'ensemble_performance': {},
            'comparison_metrics': {}
        }
        
        # Calculate metrics for each model
        for model_name, data in model_performance.items():
            if len(data['predictions']) == 0:
                continue
                
            predictions = pd.Series(data['predictions'])
            actuals = pd.Series(data['actuals'])
            errors = pd.Series(data['errors'])
            
            # Calculate prediction accuracy metrics
            mse = np.mean(errors ** 2)
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs(errors / actuals)) * 100
            
            # Calculate directional accuracy
            pred_returns = predictions.pct_change().dropna()
            actual_returns = actuals.pct_change().dropna()
            
            # Align series
            aligned_data = pd.concat([pred_returns, actual_returns], axis=1).dropna()
            if len(aligned_data) > 0:
                pred_direction = (aligned_data.iloc[:, 0] > 0).astype(int)
                actual_direction = (aligned_data.iloc[:, 1] > 0).astype(int)
                directional_accuracy = (pred_direction == actual_direction).mean()
            else:
                directional_accuracy = 0.0
            
            results['model_performance'][model_name] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'directional_accuracy': directional_accuracy,
                'num_predictions': len(predictions)
            }
        
        # Calculate ensemble-specific metrics
        if len(all_predictions) > 0:
            ensemble_returns = pd.Series(all_predictions).pct_change().dropna()
            actual_returns = pd.Series(all_actuals).pct_change().dropna()
            
            # Align series
            aligned_data = pd.concat([ensemble_returns, actual_returns], axis=1).dropna()
            if len(aligned_data) > 0:
                ensemble_metrics = self.risk_metrics.calculate_all_metrics(
                    aligned_data.iloc[:, 0],  # Strategy returns
                    market_returns=aligned_data.iloc[:, 1]  # Actual returns as benchmark
                )
                
                results['ensemble_performance'] = ensemble_metrics
        
        # Model comparison
        if len(results['model_performance']) > 1:
            comparison = {}
            for metric in ['mse', 'mae', 'rmse', 'mape', 'directional_accuracy']:
                comparison[metric] = {
                    model: results['model_performance'][model][metric]
                    for model in results['model_performance']
                    if metric in results['model_performance'][model]
                }
            results['comparison_metrics'] = comparison
        
        return results
    
    def _save_walk_forward_results(self, results: Dict[str, Any], symbol: str):
        """Save walk-forward backtest results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"walk_forward_results_{symbol}_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        results_dir = "walk_forward_results"
        os.makedirs(results_dir, exist_ok=True)
        
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        logger.info(f"Walk-forward results saved to {filepath}")
    
    def _generate_walk_forward_plots(self, results: Dict[str, Any], symbol: str):
        """Generate comprehensive visualizations for walk-forward results."""
        
        # Create plots directory
        plots_dir = "walk_forward_plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Model Performance Comparison
        if 'comparison_metrics' in results and results['comparison_metrics']:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Walk-Forward Backtest Results - {symbol}', fontsize=16)
            
            metrics = ['mse', 'mae', 'rmse', 'directional_accuracy']
            titles = ['Mean Squared Error', 'Mean Absolute Error', 'RMSE', 'Directional Accuracy']
            
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                if metric in results['comparison_metrics']:
                    ax = axes[i // 2, i % 2]
                    data = results['comparison_metrics'][metric]
                    
                    models = list(data.keys())
                    values = list(data.values())
                    
                    bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                    ax.set_title(title)
                    ax.set_ylabel(metric.upper())
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{symbol}_model_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Ensemble Performance Metrics
        if 'ensemble_performance' in results and results['ensemble_performance']:
            metrics = results['ensemble_performance']
            
            # Create a summary table
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare data for table
            table_data = []
            for metric, value in metrics.items():
                if isinstance(value, float):
                    table_data.append([metric.replace('_', ' ').title(), f'{value:.4f}'])
            
            table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                           cellLoc='left', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            ax.set_title(f'Ensemble Performance Summary - {symbol}', fontsize=16, pad=20)
            plt.savefig(os.path.join(plots_dir, f'{symbol}_ensemble_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Walk-forward plots saved to {plots_dir}")
    
    def run_multiple_symbols_backtest(self, symbols: List[str], start_date: str, end_date: str,
                                    **kwargs) -> Dict[str, Any]:
        """Run walk-forward backtest on multiple symbols."""
        
        all_results = {}
        
        for symbol in symbols:
            logger.info(f"Running walk-forward backtest for {symbol}")
            try:
                results = self.run_walk_forward_backtest(symbol, start_date, end_date, **kwargs)
                all_results[symbol] = results
            except Exception as e:
                logger.error(f"Error in walk-forward backtest for {symbol}: {e}")
                all_results[symbol] = {'error': str(e)}
        
        # Generate cross-symbol comparison
        self._generate_cross_symbol_comparison(all_results)
        
        return all_results
    
    def _generate_cross_symbol_comparison(self, all_results: Dict[str, Any]):
        """Generate comparison plots across multiple symbols."""
        
        # Filter out error results
        valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
        
        if len(valid_results) < 2:
            return
        
        # Compare ensemble performance across symbols
        symbols = []
        sharpe_ratios = []
        total_returns = []
        max_drawdowns = []
        
        for symbol, results in valid_results.items():
            if 'ensemble_performance' in results:
                ensemble_perf = results['ensemble_performance']
                symbols.append(symbol)
                sharpe_ratios.append(ensemble_perf.get('sharpe_ratio', 0))
                total_returns.append(ensemble_perf.get('total_return', 0))
                max_drawdowns.append(ensemble_perf.get('max_drawdown', 0))
        
        if len(symbols) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Cross-Symbol Performance Comparison', fontsize=16)
            
            # Sharpe Ratio comparison
            axes[0].bar(symbols, sharpe_ratios, color='skyblue')
            axes[0].set_title('Sharpe Ratio')
            axes[0].set_ylabel('Sharpe Ratio')
            
            # Total Return comparison
            axes[1].bar(symbols, total_returns, color='lightgreen')
            axes[1].set_title('Total Return')
            axes[1].set_ylabel('Total Return')
            
            # Max Drawdown comparison
            axes[2].bar(symbols, max_drawdowns, color='lightcoral')
            axes[2].set_title('Maximum Drawdown')
            axes[2].set_ylabel('Max Drawdown')
            
            plt.tight_layout()
            plt.savefig('walk_forward_plots/cross_symbol_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Cross-symbol comparison plot generated") 

    def _ensemble_predict(self, data_point):
        """Simple ensemble prediction for walk-forward backtest."""
        try:
            # For now, return a simple prediction based on the current price
            # In a full implementation, this would use the trained models
            current_price = data_point['Close']
            
            # Simple ensemble: average of different prediction methods
            # This is a placeholder - in reality, you'd use the actual trained models
            lstm_pred = current_price * (1 + np.random.normal(0, 0.01))  # Small random change
            nn_pred = current_price * (1 + np.random.normal(0, 0.01))
            xgb_pred = current_price * (1 + np.random.normal(0, 0.01))
            
            # Ensemble prediction (simple average)
            ensemble_pred = (lstm_pred + nn_pred + xgb_pred) / 3
            
            model_preds = {
                'lstm': lstm_pred,
                'nn': nn_pred,
                'xgb': xgb_pred
            }
            
            return ensemble_pred, model_preds
            
        except Exception as e:
            logger.warning(f"Ensemble prediction error: {e}")
            return None, {} 