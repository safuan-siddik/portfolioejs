"""
Model Performance Monitoring Module
===================================
This module provides a comprehensive monitoring system for model performance,
drift detection, and alerting.

Classes:
- ModelPerformanceMonitor: The main class for monitoring model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json
import os
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("ModelMonitoring")

class ModelPerformanceMonitor:
    """
    A comprehensive model performance monitoring system.

    This class is responsible for tracking model performance, detecting data
    drift, and generating alerts when performance degrades.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model performance monitor.
        
        Args:
            config (Dict[str, Any]): The configuration dictionary.
        """
        self.config = config
        self.monitoring_data = {
            'lstm': {'predictions': [], 'actuals': [], 'errors': [], 'timestamps': []},
            'nn': {'predictions': [], 'actuals': [], 'errors': [], 'timestamps': []},
            'xgb': {'predictions': [], 'actuals': [], 'errors': [], 'timestamps': []},
            'ensemble': {'predictions': [], 'actuals': [], 'errors': [], 'timestamps': []}
        }
        self.performance_metrics = {}
        self.drift_indicators = {}
        self.alerts = []
        
        # Monitoring parameters
        self.alert_thresholds = config.get('alert_thresholds', {
            'error_increase': 0.5,  # 50% increase in error
            'accuracy_drop': 0.1,   # 10% drop in accuracy
            'drift_threshold': 0.05, # 5% drift threshold
            'consecutive_failures': 5  # 5 consecutive prediction failures
        })
        
        # Performance windows
        self.short_window = config.get('short_window', 10)
        self.medium_window = config.get('medium_window', 30)
        self.long_window = config.get('long_window', 100)
        
    def update_predictions(self, model_name: str, prediction: float, actual: float,
                           timestamp: Optional[datetime] = None):
        """
        Update the monitoring data with a new prediction.
        
        Args:
            model_name (str): The name of the model.
            prediction (float): The model's prediction.
            actual (float): The actual value.
            timestamp (Optional[datetime]): The timestamp of the prediction.
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if model_name in self.monitoring_data:
            error = abs(prediction - actual) / actual if actual != 0 else 0
            
            self.monitoring_data[model_name]['predictions'].append(prediction)
            self.monitoring_data[model_name]['actuals'].append(actual)
            self.monitoring_data[model_name]['errors'].append(error)
            self.monitoring_data[model_name]['timestamps'].append(timestamp)
            
            # Keep only recent data
            max_history = self.config.get('max_history', 1000)
            if len(self.monitoring_data[model_name]['predictions']) > max_history:
                self.monitoring_data[model_name]['predictions'] = self.monitoring_data[model_name]['predictions'][-max_history:]
                self.monitoring_data[model_name]['actuals'] = self.monitoring_data[model_name]['actuals'][-max_history:]
                self.monitoring_data[model_name]['errors'] = self.monitoring_data[model_name]['errors'][-max_history:]
                self.monitoring_data[model_name]['timestamps'] = self.monitoring_data[model_name]['timestamps'][-max_history:]
    
    def calculate_performance_metrics(self, model_name: str) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for a model.
        
        Args:
            model_name (str): The name of the model.
            
        Returns:
            Dict[str, float]: A dictionary of performance metrics.
        """
        if model_name not in self.monitoring_data:
            return {}
        
        data = self.monitoring_data[model_name]
        if len(data['predictions']) == 0:
            return {}
        
        predictions = np.array(data['predictions'])
        actuals = np.array(data['actuals'])
        errors = np.array(data['errors'])
        
        # Basic metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(errors)) * 100
        
        # Directional accuracy
        pred_returns = np.diff(predictions)
        actual_returns = np.diff(actuals)
        
        if len(pred_returns) > 0:
            pred_direction = (pred_returns > 0).astype(int)
            actual_direction = (actual_returns > 0).astype(int)
            directional_accuracy = np.mean(pred_direction == actual_direction)
        else:
            directional_accuracy = 0.0
        
        # Rolling metrics
        if len(errors) >= self.short_window:
            recent_mse = mean_squared_error(actuals[-self.short_window:], predictions[-self.short_window:])
            recent_mae = mean_absolute_error(actuals[-self.short_window:], predictions[-self.short_window:])
        else:
            recent_mse = mse
            recent_mae = mae
        
        # Performance trends
        if len(errors) >= self.medium_window:
            short_errors = errors[-self.short_window:]
            medium_errors = errors[-self.medium_window:]
            error_trend = np.mean(short_errors) - np.mean(medium_errors)
        else:
            error_trend = 0.0
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'recent_mse': recent_mse,
            'recent_mae': recent_mae,
            'error_trend': error_trend,
            'total_predictions': len(predictions)
        }
        
        self.performance_metrics[model_name] = metrics
        return metrics
    
    def detect_data_drift(self, model_name: str, current_data: pd.DataFrame,
                          reference_data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect data drift between the current and reference data.
        
        Args:
            model_name (str): The name of the model.
            current_data (pd.DataFrame): The current data distribution.
            reference_data (pd.DataFrame): The reference data distribution.
            
        Returns:
            Dict[str, float]: A dictionary of drift scores.
        """
        drift_scores = {}
        
        # Statistical drift detection
        for column in current_data.columns:
            if column in reference_data.columns:
                try:
                    # Kolmogorov-Smirnov test for distribution drift
                    ks_stat, p_value = stats.ks_2samp(
                        current_data[column].dropna(),
                        reference_data[column].dropna()
                    )
                    
                    # Population Stability Index (PSI)
                    psi = self._calculate_psi(current_data[column], reference_data[column])
                    
                    drift_scores[f'{column}_ks_stat'] = ks_stat
                    drift_scores[f'{column}_ks_pvalue'] = p_value
                    drift_scores[f'{column}_psi'] = psi
                    
                except Exception as e:
                    logger.warning(f"Error calculating drift for {column}: {e}")
        
        # Feature drift summary
        if drift_scores:
            avg_psi = np.mean([v for k, v in drift_scores.items() if 'psi' in k])
            avg_ks = np.mean([v for k, v in drift_scores.items() if 'ks_stat' in k])
            
            drift_scores['avg_psi'] = avg_psi
            drift_scores['avg_ks_stat'] = avg_ks
            drift_scores['drift_detected'] = avg_psi > self.alert_thresholds['drift_threshold']
        
        self.drift_indicators[model_name] = drift_scores
        return drift_scores
    
    def _calculate_psi(self, current: pd.Series, reference: pd.Series, bins: int = 10) -> float:
        """
        Calculate the Population Stability Index (PSI).

        Args:
            current (pd.Series): The current data series.
            reference (pd.Series): The reference data series.
            bins (int): The number of bins to use.

        Returns:
            float: The PSI value.
        """
        try:
            # Create bins based on reference data
            bin_edges = np.percentile(reference.dropna(), np.linspace(0, 100, bins + 1))
            
            # Calculate histograms
            ref_hist, _ = np.histogram(reference.dropna(), bins=bin_edges)
            curr_hist, _ = np.histogram(current.dropna(), bins=bin_edges)
            
            # Normalize to probabilities
            ref_prob = ref_hist / np.sum(ref_hist)
            curr_prob = curr_hist / np.sum(curr_hist)
            
            # Calculate PSI
            psi = 0
            for i in range(len(ref_prob)):
                if ref_prob[i] > 0 and curr_prob[i] > 0:
                    psi += (curr_prob[i] - ref_prob[i]) * np.log(curr_prob[i] / ref_prob[i])
            
            return psi
            
        except Exception as e:
            logger.warning(f"Error calculating PSI: {e}")
            return 0.0
    
    def check_performance_alerts(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Check for performance alerts and generate notifications.
        
        Args:
            model_name (str): The name of the model.
            
        Returns:
            List[Dict[str, Any]]: A list of alert dictionaries.
        """
        alerts = []
        
        if model_name not in self.performance_metrics:
            return alerts
        
        metrics = self.performance_metrics[model_name]
        data = self.monitoring_data[model_name]
        
        # Check for error increase
        if len(data['errors']) >= self.medium_window:
            recent_errors = data['errors'][-self.short_window:]
            historical_errors = data['errors'][-self.medium_window:-self.short_window]
            
            if len(historical_errors) > 0:
                error_increase = (np.mean(recent_errors) - np.mean(historical_errors)) / np.mean(historical_errors)
                
                if error_increase > self.alert_thresholds['error_increase']:
                    alerts.append({
                        'type': 'error_increase',
                        'model': model_name,
                        'severity': 'high',
                        'message': f'Error increased by {error_increase:.2%}',
                        'timestamp': datetime.now(),
                        'value': error_increase
                    })
        
        # Check for accuracy drop
        if 'directional_accuracy' in metrics:
            if metrics['directional_accuracy'] < 0.4:  # Below 40% accuracy
                alerts.append({
                    'type': 'low_accuracy',
                    'model': model_name,
                    'severity': 'high',
                    'message': f'Directional accuracy is {metrics["directional_accuracy"]:.2%}',
                    'timestamp': datetime.now(),
                    'value': metrics['directional_accuracy']
                })
        
        # Check for consecutive failures
        if len(data['errors']) >= self.alert_thresholds['consecutive_failures']:
            recent_errors = data['errors'][-self.alert_thresholds['consecutive_failures']:]
            if all(error > np.mean(data['errors']) * 2 for error in recent_errors):
                alerts.append({
                    'type': 'consecutive_failures',
                    'model': model_name,
                    'severity': 'critical',
                    'message': f'{len(recent_errors)} consecutive high-error predictions',
                    'timestamp': datetime.now(),
                    'value': len(recent_errors)
                })
        
        # Check for data drift
        if model_name in self.drift_indicators:
            drift_data = self.drift_indicators[model_name]
            if 'avg_psi' in drift_data and drift_data['avg_psi'] > self.alert_thresholds['drift_threshold']:
                alerts.append({
                    'type': 'data_drift',
                    'model': model_name,
                    'severity': 'medium',
                    'message': f'Data drift detected (PSI: {drift_data["avg_psi"]:.3f})',
                    'timestamp': datetime.now(),
                    'value': drift_data['avg_psi']
                })
        
        # Store alerts
        self.alerts.extend(alerts)
        
        return alerts
    
    def generate_performance_report(self, model_name: str) -> str:
        """
        Generate a comprehensive performance report for a model.
        
        Args:
            model_name (str): The name of the model.
            
        Returns:
            str: A formatted performance report.
        """
        if model_name not in self.performance_metrics:
            return f"No performance data available for {model_name}"
        
        metrics = self.performance_metrics[model_name]
        data = self.monitoring_data[model_name]
        
        # Calculate additional metrics
        if len(data['errors']) > 0:
            error_percentiles = np.percentile(data['errors'], [25, 50, 75, 95])
            error_std = np.std(data['errors'])
        else:
            error_percentiles = [0, 0, 0, 0]
            error_std = 0
        
        report = f"""
=== {model_name.upper()} Performance Report ===

BASIC METRICS:
- Total Predictions: {metrics['total_predictions']}
- Mean Squared Error: {metrics['mse']:.6f}
- Mean Absolute Error: {metrics['mae']:.6f}
- Root Mean Squared Error: {metrics['rmse']:.6f}
- Mean Absolute Percentage Error: {metrics['mape']:.2f}%

ACCURACY METRICS:
- Directional Accuracy: {metrics['directional_accuracy']:.2%}

RECENT PERFORMANCE:
- Recent MSE: {metrics['recent_mse']:.6f}
- Recent MAE: {metrics['recent_mae']:.6f}
- Error Trend: {metrics['error_trend']:.6f}

ERROR DISTRIBUTION:
- Error Standard Deviation: {error_std:.6f}
- 25th Percentile: {error_percentiles[0]:.6f}
- Median: {error_percentiles[1]:.6f}
- 75th Percentile: {error_percentiles[2]:.6f}
- 95th Percentile: {error_percentiles[3]:.6f}

ALERTS:
"""
        
        # Add recent alerts
        recent_alerts = [alert for alert in self.alerts 
                        if alert['model'] == model_name and 
                        alert['timestamp'] > datetime.now() - timedelta(days=7)]
        
        if recent_alerts:
            for alert in recent_alerts:
                report += f"- {alert['severity'].upper()}: {alert['message']}\n"
        else:
            report += "- No recent alerts\n"
        
        return report
    
    def plot_performance_trends(self, model_name: str, save_path: Optional[str] = None):
        """
        Generate performance trend plots.
        
        Args:
            model_name (str): The name of the model.
            save_path (Optional[str]): The path to save the plots to.
        """
        if model_name not in self.monitoring_data:
            logger.warning(f"No data available for {model_name}")
            return
        
        data = self.monitoring_data[model_name]
        if len(data['predictions']) == 0:
            logger.warning(f"No predictions available for {model_name}")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name.upper()} Performance Monitoring', fontsize=16)
        
        # Plot 1: Predictions vs Actuals
        axes[0, 0].plot(data['timestamps'], data['predictions'], label='Predictions', alpha=0.7)
        axes[0, 0].plot(data['timestamps'], data['actuals'], label='Actuals', alpha=0.7)
        axes[0, 0].set_title('Predictions vs Actuals')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Error over time
        axes[0, 1].plot(data['timestamps'], data['errors'], color='red', alpha=0.7)
        axes[0, 1].set_title('Prediction Errors Over Time')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Rolling error metrics
        if len(data['errors']) >= self.short_window:
            rolling_mse = pd.Series(data['errors']).rolling(self.short_window).apply(lambda x: np.mean(x**2))
            rolling_mae = pd.Series(data['errors']).rolling(self.short_window).apply(lambda x: np.mean(np.abs(x)))
            
            axes[1, 0].plot(data['timestamps'], rolling_mse, label='Rolling MSE', color='blue')
            axes[1, 0].plot(data['timestamps'], rolling_mae, label='Rolling MAE', color='orange')
            axes[1, 0].set_title(f'Rolling Error Metrics ({self.short_window}-period)')
            axes[1, 0].set_ylabel('Error')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Error distribution
        axes[1, 1].hist(data['errors'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].set_xlabel('Error')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_monitoring_data(self, filepath: str):
        """
        Save the monitoring data to a file.

        Args:
            filepath (str): The path to save the data to.
        """
        try:
            # Convert timestamps to strings for JSON serialization
            data_to_save = {}
            for model, data in self.monitoring_data.items():
                data_to_save[model] = {
                    'predictions': data['predictions'],
                    'actuals': data['actuals'],
                    'errors': data['errors'],
                    'timestamps': [ts.isoformat() for ts in data['timestamps']]
                }
            
            with open(filepath, 'w') as f:
                json.dump({
                    'monitoring_data': data_to_save,
                    'performance_metrics': self.performance_metrics,
                    'drift_indicators': self.drift_indicators,
                    'alerts': [alert for alert in self.alerts if isinstance(alert['timestamp'], str)]
                }, f, indent=4)
            
            logger.info(f"Monitoring data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    def load_monitoring_data(self, filepath: str):
        """
        Load the monitoring data from a file.

        Args:
            filepath (str): The path to load the data from.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert timestamp strings back to datetime objects
            for model, model_data in data['monitoring_data'].items():
                self.monitoring_data[model] = {
                    'predictions': model_data['predictions'],
                    'actuals': model_data['actuals'],
                    'errors': model_data['errors'],
                    'timestamps': [datetime.fromisoformat(ts) for ts in model_data['timestamps']]
                }
            
            self.performance_metrics = data.get('performance_metrics', {})
            self.drift_indicators = data.get('drift_indicators', {})
            
            # Convert alert timestamps
            alerts = data.get('alerts', [])
            for alert in alerts:
                alert['timestamp'] = datetime.fromisoformat(alert['timestamp'])
            self.alerts = alerts
            
            logger.info(f"Monitoring data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading monitoring data: {e}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for all models.

        Returns:
            Dict[str, Any]: A dictionary of summary statistics.
        """
        summary = {
            'total_alerts': len(self.alerts),
            'recent_alerts': len([a for a in self.alerts if a['timestamp'] > datetime.now() - timedelta(days=7)]),
            'models': {}
        }
        
        for model_name in self.monitoring_data:
            if model_name in self.performance_metrics:
                metrics = self.performance_metrics[model_name]
                summary['models'][model_name] = {
                    'total_predictions': metrics['total_predictions'],
                    'current_mae': metrics['mae'],
                    'current_accuracy': metrics['directional_accuracy'],
                    'error_trend': metrics['error_trend']
                }
        
        return summary 