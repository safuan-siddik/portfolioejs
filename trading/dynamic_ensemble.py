"""
Dynamic Ensemble Weighting Module
=================================
This module provides an advanced ensemble weighting system that adapts to model
performance and market conditions. It is designed to improve the robustness
of the trading bot by dynamically adjusting the weights of the different models.

Classes:
- DynamicEnsemble: The main class that manages the ensemble weighting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("DynamicEnsemble")

class DynamicEnsemble:
    """
    A dynamic ensemble weighting system that adapts to model performance and market conditions.

    This class is responsible for calculating the weights of the different models
    in the ensemble, based on their recent performance and the current market
    conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dynamic ensemble.
        
        Args:
            config (Dict[str, Any]): The configuration dictionary.
        """
        self.config = config
        self.model_weights = {'lstm': 0.33, 'nn': 0.33, 'xgb': 0.34}
        self.performance_history = {
            'lstm': [],
            'nn': [],
            'xgb': []
        }
        self.market_conditions = []
        self.weight_history = []
        
        # Performance tracking parameters
        self.lookback_period = config.get('ensemble_lookback', 30)
        self.min_weight = config.get('min_weight', 0.1)
        self.max_weight = config.get('max_weight', 0.6)
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        
    def update_weights(self, predictions: Dict[str, float], actual_price: float,
                      market_conditions: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Update the ensemble weights based on recent performance and market conditions.
        
        Args:
            predictions (Dict[str, float]): A dictionary of model predictions.
            actual_price (float): The actual price for performance calculation.
            market_conditions (Optional[Dict[str, float]]): A dictionary of market condition indicators.
            
        Returns:
            Dict[str, float]: The updated weights dictionary.
        """
        # Calculate prediction errors
        errors = {}
        for model, pred in predictions.items():
            if model in self.model_weights:
                error = abs(pred - actual_price) / actual_price
                errors[model] = error
                self.performance_history[model].append(error)
                
                # Keep only recent history
                if len(self.performance_history[model]) > self.lookback_period:
                    self.performance_history[model] = self.performance_history[model][-self.lookback_period:]
        
        # Store market conditions
        if market_conditions:
            self.market_conditions.append(market_conditions)
            if len(self.market_conditions) > self.lookback_period:
                self.market_conditions = self.market_conditions[-self.lookback_period:]
        
        # Calculate new weights
        new_weights = self._calculate_adaptive_weights(errors, market_conditions)
        
        # Smooth weight transitions
        smoothed_weights = self._smooth_weights(new_weights)
        
        # Update weights
        self.model_weights = smoothed_weights
        self.weight_history.append(smoothed_weights.copy())
        
        return smoothed_weights
    
    def _calculate_adaptive_weights(self, errors: Dict[str, float],
                                  market_conditions: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate the adaptive weights based on performance and market conditions.

        Args:
            errors (Dict[str, float]): A dictionary of prediction errors.
            market_conditions (Optional[Dict[str, float]]): A dictionary of market condition indicators.

        Returns:
            Dict[str, float]: The calculated adaptive weights.
        """
        
        # Base weights from inverse error (better performance = higher weight)
        inverse_errors = {}
        total_inverse = 0
        
        for model, error in errors.items():
            if len(self.performance_history[model]) > 0:
                # Use recent average error
                recent_errors = self.performance_history[model][-10:]  # Last 10 predictions
                avg_error = np.mean(recent_errors)
                inverse_errors[model] = 1 / (avg_error + 1e-8)  # Avoid division by zero
                total_inverse += inverse_errors[model]
            else:
                inverse_errors[model] = 1.0
                total_inverse += 1.0
        
        # Normalize to get base weights
        base_weights = {}
        for model in inverse_errors:
            base_weights[model] = inverse_errors[model] / total_inverse
        
        # Apply market condition adjustments
        if market_conditions and len(self.market_conditions) > 5:
            adjusted_weights = self._apply_market_adjustments(base_weights, market_conditions)
        else:
            adjusted_weights = base_weights
        
        # Apply constraints
        constrained_weights = self._apply_weight_constraints(adjusted_weights)
        
        return constrained_weights
    
    def _apply_market_adjustments(self, base_weights: Dict[str, float],
                                market_conditions: Dict[str, float]) -> Dict[str, float]:
        """
        Apply market condition-based adjustments to the weights.

        Args:
            base_weights (Dict[str, float]): The base weights calculated from performance.
            market_conditions (Dict[str, float]): A dictionary of market condition indicators.

        Returns:
            Dict[str, float]: The adjusted weights.
        """
        
        adjusted_weights = base_weights.copy()
        
        # Volatility adjustment
        volatility = market_conditions.get('volatility', 0.2)
        if volatility > 0.3:  # High volatility
            # Favor more stable models (LSTM and NN)
            adjusted_weights['lstm'] *= 1.2
            adjusted_weights['nn'] *= 1.1
            adjusted_weights['xgb'] *= 0.8
        elif volatility < 0.1:  # Low volatility
            # Favor XGBoost for trend following
            adjusted_weights['xgb'] *= 1.2
            adjusted_weights['lstm'] *= 0.9
            adjusted_weights['nn'] *= 0.9
        
        # Trend strength adjustment
        trend_strength = market_conditions.get('trend_strength', 0)
        if abs(trend_strength) > 0.1:  # Strong trend
            # Favor LSTM for trend following
            adjusted_weights['lstm'] *= 1.15
            adjusted_weights['xgb'] *= 1.05
            adjusted_weights['nn'] *= 0.9
        else:  # Weak trend
            # Favor NN for pattern recognition
            adjusted_weights['nn'] *= 1.15
            adjusted_weights['lstm'] *= 0.9
            adjusted_weights['xgb'] *= 0.95
        
        # Volume adjustment
        volume_ratio = market_conditions.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:  # High volume
            # Favor XGBoost for momentum
            adjusted_weights['xgb'] *= 1.1
            adjusted_weights['lstm'] *= 0.95
            adjusted_weights['nn'] *= 0.95
        
        return adjusted_weights
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply minimum and maximum weight constraints.

        Args:
            weights (Dict[str, float]): The weights to constrain.

        Returns:
            Dict[str, float]: The constrained weights.
        """
        
        constrained_weights = weights.copy()
        
        # Normalize weights
        total_weight = sum(constrained_weights.values())
        for model in constrained_weights:
            constrained_weights[model] /= total_weight
        
        # Apply min/max constraints
        for model in constrained_weights:
            if constrained_weights[model] < self.min_weight:
                constrained_weights[model] = self.min_weight
            elif constrained_weights[model] > self.max_weight:
                constrained_weights[model] = self.max_weight
        
        # Renormalize after constraints
        total_weight = sum(constrained_weights.values())
        for model in constrained_weights:
            constrained_weights[model] /= total_weight
        
        return constrained_weights
    
    def _smooth_weights(self, new_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Smooth the weight transitions to avoid sudden changes.

        Args:
            new_weights (Dict[str, float]): The new weights to smooth.

        Returns:
            Dict[str, float]: The smoothed weights.
        """
        
        if len(self.weight_history) == 0:
            return new_weights
        
        current_weights = self.weight_history[-1]
        smoothed_weights = {}
        
        for model in new_weights:
            current_weight = current_weights.get(model, 0.33)
            target_weight = new_weights[model]
            
            # Exponential smoothing
            smoothed_weight = (1 - self.adaptation_rate) * current_weight + self.adaptation_rate * target_weight
            smoothed_weights[model] = smoothed_weight
        
        return smoothed_weights
    
    def get_ensemble_prediction(self, predictions: Dict[str, float]) -> float:
        """
        Get the weighted ensemble prediction.
        
        Args:
            predictions (Dict[str, float]): A dictionary of model predictions.
            
        Returns:
            float: The weighted ensemble prediction.
        """
        weighted_sum = 0
        total_weight = 0
        
        for model, pred in predictions.items():
            if model in self.model_weights:
                weight = self.model_weights[model]
                weighted_sum += pred * weight
                total_weight += weight
        
        if total_weight == 0:
            # Fallback to simple average
            return np.mean(list(predictions.values()))
        
        return weighted_sum / total_weight
    
    def get_model_confidence(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate confidence scores for each model based on recent performance.
        
        Args:
            predictions (Dict[str, float]): A dictionary of model predictions.
            
        Returns:
            Dict[str, float]: A dictionary of confidence scores.
        """
        confidence_scores = {}
        
        for model in predictions:
            if model in self.performance_history and len(self.performance_history[model]) > 0:
                # Calculate confidence based on recent performance stability
                recent_errors = self.performance_history[model][-10:]
                error_std = np.std(recent_errors)
                error_mean = np.mean(recent_errors)
                
                # Higher confidence for lower and more stable errors
                confidence = 1 / (1 + error_mean + error_std)
                confidence_scores[model] = confidence
            else:
                confidence_scores[model] = 0.5  # Default confidence
        
        return confidence_scores
    
    def get_ensemble_confidence(self, predictions: Dict[str, float]) -> float:
        """
        Calculate the overall ensemble confidence.
        
        Args:
            predictions (Dict[str, float]): A dictionary of model predictions.
            
        Returns:
            float: The overall ensemble confidence score.
        """
        # Calculate prediction dispersion
        pred_values = list(predictions.values())
        pred_std = np.std(pred_values)
        pred_mean = np.mean(pred_values)
        
        # Lower dispersion = higher confidence
        dispersion_confidence = 1 / (1 + pred_std / pred_mean)
        
        # Calculate model agreement
        model_confidence = self.get_model_confidence(predictions)
        avg_model_confidence = np.mean(list(model_confidence.values()))
        
        # Combine dispersion and model confidence
        ensemble_confidence = 0.7 * dispersion_confidence + 0.3 * avg_model_confidence
        
        return ensemble_confidence
    
    def get_weight_analysis(self) -> Dict[str, Any]:
        """
        Get an analysis of the weight evolution and model performance.
        
        Returns:
            Dict[str, Any]: A dictionary containing the weight analysis.
        """
        if len(self.weight_history) == 0:
            return {}
        
        analysis = {
            'current_weights': self.model_weights.copy(),
            'weight_evolution': {},
            'model_performance': {},
            'weight_stability': {}
        }
        
        # Analyze weight evolution
        weight_df = pd.DataFrame(self.weight_history)
        for model in self.model_weights:
            analysis['weight_evolution'][model] = {
                'mean': weight_df[model].mean(),
                'std': weight_df[model].std(),
                'min': weight_df[model].min(),
                'max': weight_df[model].max(),
                'trend': self._calculate_trend(weight_df[model])
            }
        
        # Analyze model performance
        for model, errors in self.performance_history.items():
            if len(errors) > 0:
                analysis['model_performance'][model] = {
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'recent_performance': np.mean(errors[-10:]) if len(errors) >= 10 else np.mean(errors)
                }
        
        # Calculate weight stability
        for model in self.model_weights:
            if model in weight_df.columns:
                weight_series = weight_df[model]
                stability = 1 - (weight_series.std() / weight_series.mean())
                analysis['weight_stability'][model] = max(0, stability)
        
        return analysis
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """
        Calculate the trend direction of a series.

        Args:
            series (pd.Series): The series to analyze.

        Returns:
            str: The trend direction ('increasing', 'decreasing', or 'stable').
        """
        if len(series) < 2:
            return "stable"
        
        # Simple linear trend
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        if slope > 0.001:
            return "increasing"
        elif slope < -0.001:
            return "decreasing"
        else:
            return "stable"
    
    def reset_weights(self):
        """Reset the weights to an equal distribution."""
        self.model_weights = {'lstm': 0.33, 'nn': 0.33, 'xgb': 0.34}
        self.weight_history = []
        logger.info("Ensemble weights reset to equal distribution")
    
    def set_fixed_weights(self, weights: Dict[str, float]):
        """
        Set fixed weights, which disables dynamic adaptation.
        
        Args:
            weights (Dict[str, float]): A dictionary of fixed weights.
        """
        # Normalize weights
        total = sum(weights.values())
        self.model_weights = {k: v/total for k, v in weights.items()}
        logger.info(f"Fixed weights set: {self.model_weights}")
    
    def get_weight_summary(self) -> str:
        """
        Get a summary of the current weights and performance.
        
        Returns:
            str: A formatted summary string.
        """
        analysis = self.get_weight_analysis()
        
        summary = f"""
=== Dynamic Ensemble Summary ===

Current Weights:
- LSTM: {self.model_weights['lstm']:.3f}
- NN: {self.model_weights['nn']:.3f}
- XGBoost: {self.model_weights['xgb']:.3f}

Weight Stability:
"""
        
        if 'weight_stability' in analysis:
            for model, stability in analysis['weight_stability'].items():
                summary += f"- {model.upper()}: {stability:.3f}\n"
        
        summary += "\nModel Performance (Recent):\n"
        if 'model_performance' in analysis:
            for model, perf in analysis['model_performance'].items():
                summary += f"- {model.upper()}: {perf['recent_performance']:.4f} (mean error)\n"
        
        return summary 