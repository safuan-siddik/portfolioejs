"""
Advanced Risk Metrics Module
============================
Comprehensive risk and performance metrics for trading strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger("RiskMetrics")

class RiskMetrics:
    """
    Advanced risk and performance metrics for trading strategies.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate daily returns from price series."""
        return prices.pct_change().dropna()
    
    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns from price series."""
        return np.log(prices / prices.shift(1)).dropna()
    
    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns."""
        return (1 + returns).cumprod()
    
    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            returns: Series of returns
            annualize: Whether to annualize the volatility
            
        Returns:
            Volatility as float
        """
        volatility = returns.std()
        if annualize:
            volatility *= np.sqrt(252)  # Assuming daily data
        return volatility
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (uses instance default if None)
            
        Returns:
            Sharpe ratio as float
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std()
        return sharpe * np.sqrt(252)  # Annualize
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (uses instance default if None)
            
        Returns:
            Sortino ratio as float
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = excess_returns.mean() / downside_returns.std()
        return sortino * np.sqrt(252)  # Annualize
    
    def calculate_calmar_ratio(self, returns: pd.Series, lookback_period: int = 252) -> float:
        """
        Calculate Calmar ratio (annual return / maximum drawdown).
        
        Args:
            returns: Series of returns
            lookback_period: Period for annual return calculation
            
        Returns:
            Calmar ratio as float
        """
        cumulative_returns = self.calculate_cumulative_returns(returns)
        max_drawdown = self.calculate_max_drawdown(cumulative_returns)
        
        if max_drawdown == 0:
            return 0.0
        
        annual_return = (cumulative_returns.iloc[-1] ** (252 / len(returns))) - 1
        calmar = annual_return / abs(max_drawdown)
        
        return calmar
    
    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            cumulative_returns: Series of cumulative returns
            
        Returns:
            Maximum drawdown as float (negative value)
        """
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()
    
    def calculate_value_at_risk(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR as float (negative value)
        """
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_conditional_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.05 for 95% CVaR)
            
        Returns:
            CVaR as float (negative value)
        """
        var = self.calculate_value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_beta(self, strategy_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta relative to market.
        
        Args:
            strategy_returns: Strategy returns
            market_returns: Market returns
            
        Returns:
            Beta as float
        """
        # Align the series
        aligned_data = pd.concat([strategy_returns, market_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 0.0
        
        strategy_ret = aligned_data.iloc[:, 0]
        market_ret = aligned_data.iloc[:, 1]
        
        covariance = np.cov(strategy_ret, market_ret)[0, 1]
        market_variance = np.var(market_ret)
        
        if market_variance == 0:
            return 0.0
        
        return covariance / market_variance
    
    def calculate_alpha(self, strategy_returns: pd.Series, market_returns: pd.Series, 
                       risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Jensen's alpha.
        
        Args:
            strategy_returns: Strategy returns
            market_returns: Market returns
            risk_free_rate: Risk-free rate (uses instance default if None)
            
        Returns:
            Alpha as float
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        beta = self.calculate_beta(strategy_returns, market_returns)
        strategy_mean = strategy_returns.mean() * 252  # Annualize
        market_mean = market_returns.mean() * 252  # Annualize
        
        alpha = strategy_mean - (risk_free_rate + beta * (market_mean - risk_free_rate))
        return alpha
    
    def calculate_information_ratio(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate Information ratio.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio as float
        """
        # Align the series
        aligned_data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 0.0
        
        strategy_ret = aligned_data.iloc[:, 0]
        benchmark_ret = aligned_data.iloc[:, 1]
        
        excess_returns = strategy_ret - benchmark_ret
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        information_ratio = excess_returns.mean() / tracking_error
        return information_ratio * np.sqrt(252)  # Annualize
    
    def calculate_ulcer_index(self, cumulative_returns: pd.Series) -> float:
        """
        Calculate Ulcer Index (measure of downside risk).
        
        Args:
            cumulative_returns: Series of cumulative returns
            
        Returns:
            Ulcer Index as float
        """
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        squared_drawdown = drawdown ** 2
        return np.sqrt(squared_drawdown.mean())
    
    def calculate_gain_to_pain_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Gain-to-Pain ratio.
        
        Args:
            returns: Series of returns
            
        Returns:
            Gain-to-Pain ratio as float
        """
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return gains if gains > 0 else 0.0
        
        return gains / losses
    
    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """
        Calculate Profit Factor.
        
        Args:
            returns: Series of returns
            
        Returns:
            Profit factor as float
        """
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return gross_profit if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """
        Calculate win rate.
        
        Args:
            returns: Series of returns
            
        Returns:
            Win rate as float (0-1)
        """
        if len(returns) == 0:
            return 0.0
        
        return (returns > 0).sum() / len(returns)
    
    def calculate_avg_win_loss_ratio(self, returns: pd.Series) -> float:
        """
        Calculate average win/loss ratio.
        
        Args:
            returns: Series of returns
            
        Returns:
            Average win/loss ratio as float
        """
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        if avg_loss == 0:
            return avg_win if avg_win > 0 else 0.0
        
        return avg_win / avg_loss
    
    def calculate_kelly_criterion(self, returns: pd.Series) -> float:
        """
        Calculate Kelly Criterion optimal position size.
        
        Args:
            returns: Series of returns
            
        Returns:
            Kelly fraction as float (0-1)
        """
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        win_rate = len(wins) / len(returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        if avg_loss == 0:
            return 0.0
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return max(0, min(1, kelly))  # Clamp between 0 and 1
    
    def calculate_all_metrics(self, returns: pd.Series, cumulative_returns: Optional[pd.Series] = None,
                            market_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate all risk and performance metrics.
        
        Args:
            returns: Series of returns
            cumulative_returns: Series of cumulative returns (calculated if None)
            market_returns: Market returns for beta/alpha calculations
            
        Returns:
            Dictionary of all metrics
        """
        if cumulative_returns is None:
            cumulative_returns = self.calculate_cumulative_returns(returns)
        
        metrics = {
            'total_return': cumulative_returns.iloc[-1] - 1,
            'annualized_return': (cumulative_returns.iloc[-1] ** (252 / len(returns))) - 1,
            'volatility': self.calculate_volatility(returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'calmar_ratio': self.calculate_calmar_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(cumulative_returns),
            'var_95': self.calculate_value_at_risk(returns, 0.05),
            'cvar_95': self.calculate_conditional_var(returns, 0.05),
            'ulcer_index': self.calculate_ulcer_index(cumulative_returns),
            'gain_to_pain_ratio': self.calculate_gain_to_pain_ratio(returns),
            'profit_factor': self.calculate_profit_factor(returns),
            'win_rate': self.calculate_win_rate(returns),
            'avg_win_loss_ratio': self.calculate_avg_win_loss_ratio(returns),
            'kelly_criterion': self.calculate_kelly_criterion(returns)
        }
        
        if market_returns is not None:
            metrics.update({
                'beta': self.calculate_beta(returns, market_returns),
                'alpha': self.calculate_alpha(returns, market_returns),
                'information_ratio': self.calculate_information_ratio(returns, market_returns)
            })
        
        return metrics
    
    def generate_risk_report(self, returns: pd.Series, strategy_name: str = "Strategy") -> str:
        """
        Generate a comprehensive risk report.
        
        Args:
            returns: Series of returns
            strategy_name: Name of the strategy
            
        Returns:
            Formatted risk report as string
        """
        metrics = self.calculate_all_metrics(returns)
        
        report = f"""
=== {strategy_name} Risk Report ===

RETURNS:
- Total Return: {metrics['total_return']:.2%}
- Annualized Return: {metrics['annualized_return']:.2%}

RISK METRICS:
- Volatility: {metrics['volatility']:.2%}
- Maximum Drawdown: {metrics['max_drawdown']:.2%}
- VaR (95%): {metrics['var_95']:.2%}
- CVaR (95%): {metrics['cvar_95']:.2%}
- Ulcer Index: {metrics['ulcer_index']:.4f}

RISK-ADJUSTED RETURNS:
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
- Sortino Ratio: {metrics['sortino_ratio']:.2f}
- Calmar Ratio: {metrics['calmar_ratio']:.2f}

TRADING METRICS:
- Win Rate: {metrics['win_rate']:.2%}
- Profit Factor: {metrics['profit_factor']:.2f}
- Gain-to-Pain Ratio: {metrics['gain_to_pain_ratio']:.2f}
- Avg Win/Loss Ratio: {metrics['avg_win_loss_ratio']:.2f}
- Kelly Criterion: {metrics['kelly_criterion']:.2%}
"""
        
        return report 