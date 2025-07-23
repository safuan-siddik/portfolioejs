#!/usr/bin/env python3
"""
Comprehensive Trading Bot Backtest
==================================
A unified backtest that combines all advanced techniques:
- Ensemble predictions (LSTM, Neural Network, XGBoost)
- Advanced entry/exit conditions
- Dynamic position sizing
- Market regime detection
- Trailing stops
- Risk management
- Hyperparameter optimization
- Walk-forward validation
- Model monitoring
- Feature selection
- Adaptive parameters
"""

import json
import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tradingbot import TradingBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_backtest.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_comprehensive_config():
    """Create a comprehensive configuration with all advanced techniques."""
    
    config = {
        # Core trading parameters
        "symbols": ["AAPL", "MSFT", "GOOGL"],  # Multiple symbols for diversification
        "lookback_days": 730,  # 2 years for robust model training
        "test_size": 0.15,     # 15% test size for more training data
        "feature_columns": ["Open", "High", "Low", "Close", "Volume"],
        "target_column": "Close",
        "sequence_length": 30,  # Longer sequence for better pattern recognition
        "training_epochs": 50,  # More epochs for better convergence
        "batch_size": 32,       # Larger batch size for stability
        "learning_rate": 0.005, # Lower learning rate for better convergence
        "hidden_size": 64,      # Larger network for better capacity
        "model_type": "ensemble",
        
        # Capital and position management - OPTIMIZED FOR BETTER RETURNS
        "initial_capital": 10000,  # Fixed to match trading bot's actual capital
        "position_size": 0.25,     # INCREASED from 0.15 to 0.25 for better returns
        "stop_loss_pct": 0.04,     # INCREASED from 0.025 to 0.04 to reduce premature exits
        "take_profit_pct": 0.12,   # INCREASED from 0.10 to 0.12 for better profit capture
        "trailing_stop_pct": 0.035, # INCREASED from 0.025 to 0.035 to reduce premature exits
        "max_holding_period": 45,   # INCREASED from 30 to 45 days for longer trends
        
        # Entry/exit thresholds - OPTIMIZED FOR BETTER PERFORMANCE
        "min_profit_threshold": 0.01,  # REDUCED from 0.015 to 0.01
        "momentum_threshold": 0.005,   # REDUCED from 0.008 to 0.005
        "volume_threshold": 0.6,       # REDUCED from 0.8 to 0.6
        "volatility_threshold": 0.6,   # REDUCED from 0.8 to 0.6
        "required_conditions": 2,      # REDUCED from 4 to 2 for easier entry
        "prediction_threshold": 0.002, # REDUCED from 0.004 to 0.002 for easier entry
        "rsi_threshold": 70,           # INCREASED from 65 to 70 for less restrictive entry
        
        # File paths
        "models_dir": "models",
        "data_dir": "data",
        
        # ALL ADVANCED FEATURES ENABLED
        "enable_hyperparameter_optimization": True,
        "enable_walk_forward_backtest": True,
        "enable_dynamic_ensemble": True,
        "enable_advanced_features": True,
        "enable_model_monitoring": True,
        "enable_risk_metrics": True,
        "enable_feature_selection": True,
        "enable_adaptive_parameters": True,
        
        # Enhanced ensemble parameters - OPTIMIZED
        "ensemble_lookback": 60,       # Longer lookback for better weighting
        "min_weight": 0.05,            # Lower minimum weight
        "max_weight": 0.7,             # Higher maximum weight
        "adaptation_rate": 0.15,       # Faster adaptation
        
        # Advanced risk management - OPTIMIZED
        "max_portfolio_risk": 0.025,   # INCREASED from 0.02 to 0.025
        "max_correlation": 0.75,       # INCREASED from 0.7 to 0.75
        "max_positions": 6,            # INCREASED from 5 to 6
        "min_risk_reward": 2.0,        # REDUCED from 2.5 to 2.0 for easier entry
        
        # Advanced alerting
        "alert_thresholds": {
            "error_increase": 0.3,
            "accuracy_drop": 0.05,
            "drift_threshold": 0.03,
            "consecutive_failures": 3
        },
        
        # Technical analysis windows - OPTIMIZED
        "short_window": 5,
        "medium_window": 20,
        "long_window": 60,
        "max_history": 2000,
        
        # Entry conditions - OPTIMIZED FOR BETTER PERFORMANCE
        "entry_conditions": {
            "min_technical_score": 0.4,      # REDUCED from 0.6 to 0.4
            "min_momentum_score": 0.3,       # REDUCED from 0.5 to 0.3
            "min_volume_score": 0.5,         # REDUCED from 0.7 to 0.5
            "min_prediction_confidence": 0.5, # REDUCED from 0.7 to 0.5
            "max_volatility_score": 0.6,     # INCREASED from 0.4 to 0.6
            "min_trend_strength": 0.4        # REDUCED from 0.6 to 0.4
        },
        
        # Exit conditions - OPTIMIZED FOR BETTER PERFORMANCE
        "exit_conditions": {
            "profit_target": 0.15,           # INCREASED from 0.12 to 0.15
            "stop_loss": 0.04,               # INCREASED from 0.025 to 0.04
            "trailing_stop": 0.035,          # INCREASED from 0.02 to 0.035
            "max_holding_days": 45,          # INCREASED from 25 to 45
            "momentum_reversal": 0.025,      # INCREASED from 0.02 to 0.025
            "volume_drop": 0.6               # INCREASED from 0.5 to 0.6
        }
    }
    return config

def save_backtest_results(results, config, timestamp):
    """Save backtest results to a dedicated results file."""
    
    # Create results directory if it doesn't exist
    results_dir = "backtest_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create comprehensive results file
    results_filename = f"{results_dir}/comprehensive_backtest_results_{timestamp}.json"
    
    # Prepare comprehensive results data
    comprehensive_results = {
        "backtest_info": {
            "backtest_id": f"comprehensive_backtest_{timestamp}",
            "timestamp": timestamp,
            "start_date": config.get("start_date", "2023-07-01"),
            "end_date": config.get("end_date", "2024-12-31"),
            "initial_capital": config.get("initial_capital", 50000),
            "final_capital": results.get("final_capital", config.get("initial_capital", 50000)),
            "total_portfolio_value": results.get("final_capital", config.get("initial_capital", 50000)) + sum(
                pos.get("shares", 0) * pos.get("entry_price", 0) 
                for pos in results.get("positions", {}).values()
            ),
            "total_return_pct": ((results.get("final_capital", config.get("initial_capital", 50000)) + sum(
                pos.get("shares", 0) * pos.get("entry_price", 0) 
                for pos in results.get("positions", {}).values()
            ) - config.get("initial_capital", 50000)) / config.get("initial_capital", 50000)) * 100,
            "symbols_traded": config.get("symbols", ["AAPL", "MSFT", "GOOGL"]),
            "strategy_type": "Comprehensive Ensemble with Advanced Features"
        },
        "portfolio_summary": {
            "cash": results.get("cash", 0),
            "positions": results.get("positions", {}),
            "total_positions_value": sum(
                pos.get("shares", 0) * pos.get("entry_price", 0) 
                for pos in results.get("positions", {}).values()
            )
        },
        "trades": results.get("trades", []),
        "portfolio_history": results.get("portfolio_history", []),
        "drawdown_data": results.get("drawdown_data", []),
        "monthly_returns": results.get("monthly_returns", {}),
        "symbol_performance": results.get("symbol_performance", {}),
        "win_loss_streaks": results.get("win_loss_streaks", []),
        "trade_action_counts": results.get("trade_action_counts", {}),
        "peak_value": results.get("peak_value", 0),
        "performance_metrics": results.get("metrics", {}),
        "configuration": {
            "position_size": config.get("position_size", 0.25),
            "stop_loss_pct": config.get("stop_loss_pct", 0.04),
            "take_profit_pct": config.get("take_profit_pct", 0.12),
            "trailing_stop_pct": config.get("trailing_stop_pct", 0.035),
            "max_holding_period": config.get("max_holding_period", 45),
            "prediction_threshold": config.get("prediction_threshold", 0.002),
            "rsi_threshold": config.get("rsi_threshold", 70),
            "required_conditions": config.get("required_conditions", 2)
        },
        "advanced_features": {
            "hyperparameter_optimization": config.get("enable_hyperparameter_optimization", True),
            "walk_forward_backtest": config.get("enable_walk_forward_backtest", True),
            "dynamic_ensemble": config.get("enable_dynamic_ensemble", True),
            "advanced_features": config.get("enable_advanced_features", True),
            "model_monitoring": config.get("enable_model_monitoring", True),
            "risk_metrics": config.get("enable_risk_metrics", True),
            "feature_selection": config.get("enable_feature_selection", True),
            "adaptive_parameters": config.get("enable_adaptive_parameters", True)
        }
    }
    
    # Save to JSON file
    with open(results_filename, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    logger.info(f"Comprehensive backtest results saved to: {results_filename}")
    return results_filename

def print_backtest_summary(results, config):
    """Print a clean summary of the backtest results."""
    
    initial_capital = config.get("initial_capital", 50000)
    final_capital = results.get("final_capital", initial_capital)
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    total_trades = len(results.get("trades", []))
    
    # Calculate win rate
    trades = results.get("trades", [])
    winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Get metrics
    metrics = results.get("metrics", {})
    
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE BACKTEST SUMMARY")
    print("=" * 80)
    
    # Calculate total portfolio value (cash + positions)
    total_portfolio_value = final_capital + sum(
        pos.get("shares", 0) * pos.get("entry_price", 0) 
        for pos in results.get("positions", {}).values()
    )
    total_return = ((total_portfolio_value - initial_capital) / initial_capital) * 100
    
    print(f"📊 PERFORMANCE OVERVIEW:")
    print(f"   💰 Initial Capital: ${initial_capital:,.2f}")
    print(f"   💰 Final Capital:   ${final_capital:,.2f}")
    print(f"   💼 Total Portfolio: ${total_portfolio_value:,.2f}")
    print(f"   📈 Total Return:    {total_return:+.2f}%")
    print(f"   📈 Annualized:      {metrics.get('annualized_return', 0)*100:.2f}%")
    
    print(f"\n📈 TRADING STATISTICS:")
    print(f"   🔄 Total Trades:    {total_trades}")
    print(f"   ✅ Winning Trades:  {winning_trades}")
    print(f"   ❌ Losing Trades:   {total_trades - winning_trades}")
    print(f"   🎯 Win Rate:        {win_rate:.1f}%")
    print(f"   💵 Avg Profit/Trade: ${metrics.get('avg_profit_per_trade', 0):+.2f}")
    print(f"   ⏱️  Avg Duration:    {metrics.get('avg_trade_duration', 0):.1f} days")
    
    print(f"\n📊 RISK METRICS:")
    print(f"   📉 Max Drawdown:    {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"   📊 Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"   📊 Sortino Ratio:   {metrics.get('sortino_ratio', 0):.3f}")
    print(f"   📊 Profit Factor:   {metrics.get('profit_factor', 0):.2f}")
    
    print(f"\n🎯 STRATEGY CONFIGURATION:")
    print(f"   📈 Position Size:   {config.get('position_size', 0.25)*100}%")
    print(f"   🛡️  Stop Loss:       {config.get('stop_loss_pct', 0.04)*100}%")
    print(f"   🎯 Take Profit:     {config.get('take_profit_pct', 0.12)*100}%")
    print(f"   📊 Prediction Threshold: {config.get('prediction_threshold', 0.002)*100}%")
    print(f"   🔄 Required Conditions: {config.get('required_conditions', 2)}")
    
    print(f"\n🚀 ADVANCED FEATURES ENABLED:")
    features = [
        ("Hyperparameter Optimization", config.get("enable_hyperparameter_optimization", True)),
        ("Walk-Forward Backtest", config.get("enable_walk_forward_backtest", True)),
        ("Dynamic Ensemble", config.get("enable_dynamic_ensemble", True)),
        ("Advanced Features", config.get("enable_advanced_features", True)),
        ("Model Monitoring", config.get("enable_model_monitoring", True)),
        ("Risk Metrics", config.get("enable_risk_metrics", True)),
        ("Feature Selection", config.get("enable_feature_selection", True)),
        ("Adaptive Parameters", config.get("enable_adaptive_parameters", True))
    ]
    
    for feature, enabled in features:
        status = "✅" if enabled else "❌"
        print(f"   {status} {feature}")
    
    # Performance assessment
    print(f"\n📊 PERFORMANCE ASSESSMENT:")
    if total_return > 0:
        if total_return > 20:
            print("  🎉 EXCELLENT: Outstanding performance!")
        elif total_return > 10:
            print("  ✅ VERY GOOD: Strong positive returns!")
        elif total_return > 5:
            print("  👍 GOOD: Solid positive returns")
        else:
            print("  📈 POSITIVE: Modest but positive returns")
    else:
        print("  ⚠️  NEEDS IMPROVEMENT: Negative returns")
    
    print("=" * 80)

def run_comprehensive_backtest():
    """Run a comprehensive backtest with all advanced techniques."""
    
    print("🚀 COMPREHENSIVE TRADING BOT BACKTEST")
    print("=" * 60)
    print("📊 All Advanced Techniques Enabled:")
    print("   ✅ Ensemble Predictions (LSTM + NN + XGBoost)")
    print("   ✅ Advanced Entry/Exit Conditions")
    print("   ✅ Dynamic Position Sizing")
    print("   ✅ Market Regime Detection")
    print("   ✅ Trailing Stops & Risk Management")
    print("   ✅ Hyperparameter Optimization")
    print("   ✅ Walk-Forward Validation")
    print("   ✅ Model Monitoring & Feature Selection")
    print("   ✅ Adaptive Parameters")
    print("=" * 60)
    
    # Create comprehensive configuration
    config = create_comprehensive_config()
    
    # Add date range to config
    config["start_date"] = "2023-07-01"
    config["end_date"] = "2024-12-31"
    
    # Save config to separate file
    config_dir = "backtest_configs"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"{config_dir}/comprehensive_config_{timestamp}.json"
    
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to: {config_filename}")
    
    try:
        # Initialize trading bot
        print("🤖 Initializing Trading Bot with all advanced features...")
        bot = TradingBot(config)
        print("✅ Trading Bot initialized successfully")
        
        # Set date range for backtest
        start_date = config["start_date"]
        end_date = config["end_date"]
        
        print(f"📅 Backtest Period: {start_date} to {end_date}")
        print(f"💰 Initial Capital: ${config['initial_capital']:,}")
        print(f"📈 Symbols: {', '.join(config['symbols'])}")
        print(f"🎯 Position Size: {config['position_size']*100}%")
        print(f"🛡️  Stop Loss: {config['stop_loss_pct']*100}%")
        print(f"🎯 Take Profit: {config['take_profit_pct']*100}%")
        print("=" * 60)
        
        # Run comprehensive backtest
        print("🔄 Running comprehensive backtest with all techniques...")
        results = bot.backtest(start_date=start_date, end_date=end_date)
        
        if results is None:
            print("❌ Backtest failed to complete")
            return False
        
        # Save results to dedicated file
        results_filename = save_backtest_results(results, config, timestamp)
        
        # Print clean summary
        print_backtest_summary(results, config)
        
        print(f"\n💾 Results saved to: {results_filename}")
        print(f"📁 Configuration saved to: {config_filename}")
        print("\n🎉 Comprehensive backtest completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during comprehensive backtest: {str(e)}")
        print(f"❌ Backtest failed with error: {str(e)}")
        return False

def main():
    """Main function to run the comprehensive backtest."""
    success = run_comprehensive_backtest()
    
    if success:
        print("\n✅ Backtest completed successfully!")
        print("📊 Check the backtest_results/ directory for detailed results.")
    else:
        print("\n❌ Backtest failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 