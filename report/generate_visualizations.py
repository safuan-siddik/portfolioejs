import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import os
import json
from datetime import datetime, timedelta

def create_figures_directory():
    """Create directory for figures if it doesn't exist."""
    if not os.path.exists('figures'):
        os.makedirs('figures')

def load_backtest_data(backtest_id):
    """Load backtest data from JSON files."""
    try:
        # Use relative path from report directory to data directory
        data_file = f'../data/backtest_data/{backtest_id}_data.json'
        results_file = f'../data/backtest_results/{backtest_id}_results.json'
        
        if not os.path.exists(data_file) or not os.path.exists(results_file):
            print(f"Warning: Backtest data files not found for ID: {backtest_id}")
            return None, None
            
        with open(data_file, 'r') as f:
            data = json.load(f)
        with open(results_file, 'r') as f:
            results = json.load(f)
        return data, results
    except Exception as e:
        print(f"Error loading backtest data: {str(e)}")
        return None, None

def plot_portfolio_growth(results):
    """Plot portfolio value growth over time."""
    plt.figure(figsize=(12, 6))
    dates = pd.date_range(start=results['start_date'], end=results['end_date'], freq='B')
    portfolio_values = results['daily_returns']
    
    plt.plot(dates, portfolio_values, label='Portfolio Value', linewidth=2)
    plt.title('Portfolio Value Growth Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add key events annotations
    max_drawdown_idx = np.argmax(np.maximum.accumulate(portfolio_values) - portfolio_values)
    plt.annotate('Max Drawdown', 
                xy=(dates[max_drawdown_idx], portfolio_values[max_drawdown_idx]),
                xytext=(dates[max_drawdown_idx] + timedelta(days=30), 
                       portfolio_values[max_drawdown_idx] * 0.9),
                arrowprops=dict(facecolor='red', shrink=0.05))
    
    plt.savefig('figures/portfolio_growth.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_risk_metrics(results):
    """Plot comprehensive risk metrics dashboard."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Drawdown analysis
    portfolio_values = np.array(results['daily_returns'])
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (running_max - portfolio_values) / running_max
    ax1.plot(drawdown, label='Drawdown', color='red')
    ax1.set_title('Portfolio Drawdown')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Drawdown')
    ax1.grid(True, alpha=0.3)
    
    # Rolling volatility
    returns = pd.Series(portfolio_values).pct_change().dropna()
    rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
    ax2.plot(rolling_vol, label='Rolling Volatility', color='blue')
    ax2.set_title('Rolling Volatility (20-day)')
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Volatility')
    ax2.grid(True, alpha=0.3)
    
    # Sharpe ratio
    rolling_sharpe = (returns.rolling(window=20).mean() * np.sqrt(252)) / rolling_vol
    ax3.plot(rolling_sharpe, label='Rolling Sharpe', color='green')
    ax3.set_title('Rolling Sharpe Ratio (20-day)')
    ax3.set_xlabel('Trading Days')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.grid(True, alpha=0.3)
    
    # Risk metrics summary
    metrics = [
        ('Max Drawdown', f"{results['metrics']['max_drawdown']:.1%}"),
        ('Sharpe Ratio', f"{results['metrics']['risk_reward_ratio']:.2f}"),
        ('Sortino Ratio', f"{results['metrics']['sortino_ratio']:.2f}"),
        ('Calmar Ratio', f"{results['metrics']['calmar_ratio']:.2f}")
    ]
    ax4.axis('off')
    table = ax4.table(cellText=[[m[0], m[1]] for m in metrics],
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    plt.tight_layout()
    plt.savefig('figures/risk_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_trade_distribution(data):
    """Plot trade P&L distribution and win/loss analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # P&L distribution
    trade_profits = data['trade_profits']
    sns.histplot(trade_profits, bins=30, kde=True, ax=ax1)
    ax1.set_title('Trade P&L Distribution')
    ax1.set_xlabel('Profit/Loss ($)')
    ax1.set_ylabel('Frequency')
    
    # Win/Loss ratio
    wins = sum(1 for p in trade_profits if p > 0)
    losses = sum(1 for p in trade_profits if p < 0)
    ax2.pie([wins, losses], labels=['Wins', 'Losses'], 
            autopct='%1.1f%%', colors=['green', 'red'])
    ax2.set_title('Win/Loss Ratio')
    
    plt.tight_layout()
    plt.savefig('figures/trade_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_trade_timing(data):
    """Plot trade entry/exit timing analysis."""
    plt.figure(figsize=(12, 6))
    
    # Convert trade dates to datetime
    trade_dates = pd.to_datetime(data['trade_dates'])
    trade_profits = data['trade_profits']
    
    # Plot trade timing
    plt.scatter(trade_dates, trade_profits, 
               c=['green' if p > 0 else 'red' for p in trade_profits],
               alpha=0.6)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title('Trade Entry/Exit Timing Analysis')
    plt.xlabel('Date')
    plt.ylabel('Profit/Loss ($)')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('figures/trade_timing.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_market_regime(results):
    """Plot market regime detection and performance."""
    plt.figure(figsize=(12, 6))
    
    # Calculate market regime indicators
    portfolio_values = np.array(results['daily_returns'])
    returns = pd.Series(portfolio_values).pct_change().dropna()
    volatility = returns.rolling(window=20).std() * np.sqrt(252)
    trend = returns.rolling(window=50).mean() * np.sqrt(252)
    
    # Plot regime indicators
    plt.plot(volatility, label='Volatility', color='red', alpha=0.5)
    plt.plot(trend, label='Trend', color='blue', alpha=0.5)
    plt.title('Market Regime Detection')
    plt.xlabel('Trading Days')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('figures/market_regime.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation(results):
    """Plot asset correlation and portfolio impact."""
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    symbols = results['symbols']
    returns_data = pd.DataFrame()
    for symbol in symbols:
        returns_data[symbol] = pd.Series(results['daily_returns']).pct_change()
    
    # Plot correlation heatmap
    sns.heatmap(returns_data.corr(), annot=True, cmap='RdYlGn', center=0)
    plt.title('Asset Correlation Matrix')
    
    plt.savefig('figures/correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_accuracy(data):
    """Plot model prediction accuracy over time."""
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs predicted prices
    plt.plot(data['actual_prices'], label='Actual', color='blue')
    plt.plot(data['model_predictions'], label='Predicted', color='red', alpha=0.7)
    plt.title('Model Prediction Accuracy Over Time')
    plt.xlabel('Trading Days')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('figures/prediction_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(results):
    """Plot feature importance analysis."""
    plt.figure(figsize=(10, 6))
    
    # Get feature importance from results
    features = results['feature_importance']
    importance = results['feature_importance_values']
    
    # Plot feature importance
    plt.barh(features, importance)
    plt.title('Feature Importance Analysis')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_sensitivity(results):
    """Plot strategy parameter sensitivity analysis."""
    plt.figure(figsize=(12, 6))
    
    # Get parameter sensitivity data
    parameters = results['parameter_sensitivity']
    performance = results['parameter_performance']
    
    # Plot parameter sensitivity
    plt.plot(parameters, performance, marker='o')
    plt.title('Parameter Sensitivity Analysis')
    plt.xlabel('Parameter Value')
    plt.ylabel('Performance Metric')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('figures/parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_optimization_results(results):
    """Plot strategy optimization results."""
    plt.figure(figsize=(12, 6))
    
    # Get optimization results
    iterations = range(len(results['optimization_history']))
    performance = results['optimization_history']
    
    # Plot optimization progress
    plt.plot(iterations, performance, marker='o')
    plt.title('Strategy Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Performance Metric')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('figures/optimization_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all visualizations for the report."""
    create_figures_directory()
    
    # Load the most recent backtest results using correct relative path
    backtest_results_dir = '../data/backtest_results'
    if not os.path.exists(backtest_results_dir):
        print(f"Error: Backtest results directory not found at {backtest_results_dir}")
        return
        
    backtest_files = [f for f in os.listdir(backtest_results_dir) if f.endswith('_results.json')]
    if not backtest_files:
        print("No backtest results found in backtest_results directory!")
        print("Please run a backtest first to generate results.")
        return
    
    latest_backtest = sorted(backtest_files)[-1]
    backtest_id = latest_backtest.replace('_results.json', '')
    
    data, results = load_backtest_data(backtest_id)
    if data is None or results is None:
        print("Failed to load backtest data. Please ensure backtest was run successfully.")
        return
    
    # Generate all visualizations
    try:
        plot_portfolio_growth(results)
        plot_risk_metrics(results)
        plot_trade_distribution(data)
        plot_trade_timing(data)
        plot_market_regime(results)
        plot_correlation(results)
        plot_prediction_accuracy(data)
        plot_feature_importance(results)
        plot_parameter_sensitivity(results)
        plot_optimization_results(results)
        print("All visualizations generated successfully!")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")

if __name__ == "__main__":
    main() 