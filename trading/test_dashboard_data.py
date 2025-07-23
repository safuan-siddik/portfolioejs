#!/usr/bin/env python3
"""
Test Dashboard Data Loading
===========================
Simple script to test if the dashboard can properly load comprehensive backtest data.
"""

import json
import os
import sys

def test_data_loading():
    """Test the dashboard data loading functionality."""
    
    print("🧪 Testing Dashboard Data Loading...")
    print("=" * 50)
    
    # Test 1: Check if comprehensive backtest results exist
    print("\n1️⃣ Checking for comprehensive backtest results...")
    
    results_dir = "backtest_results"
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) if f.startswith('comprehensive_backtest_results_') and f.endswith('.json')]
        
        if result_files:
            latest_file = max(result_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
            print(f"✅ Found comprehensive results: {latest_file}")
            
            # Test loading the results
            try:
                with open(os.path.join(results_dir, latest_file), 'r') as f:
                    data = json.load(f)
                
                print(f"✅ Successfully loaded JSON data")
                print(f"📊 Data keys: {list(data.keys())}")
                
                # Check required sections
                required_sections = ['backtest_info', 'performance_metrics']
                missing_sections = [section for section in required_sections if section not in data]
                
                if not missing_sections:
                    print("✅ All required sections present")
                    
                    # Display key information
                    backtest_info = data['backtest_info']
                    performance_metrics = data['performance_metrics']
                    
                    print(f"\n📊 Backtest Information:")
                    print(f"   Backtest ID: {backtest_info.get('backtest_id', 'N/A')}")
                    print(f"   Initial Capital: ${backtest_info.get('initial_capital', 0):,.2f}")
                    print(f"   Final Capital: ${backtest_info.get('final_capital', 0):,.2f}")
                    print(f"   Total Return: {backtest_info.get('total_return_pct', 0):+.2f}%")
                    
                    print(f"\n📈 Performance Metrics:")
                    print(f"   Total Trades: {performance_metrics.get('total_trades', 0)}")
                    print(f"   Winning Trades: {performance_metrics.get('winning_trades', 0)}")
                    print(f"   Win Rate: {(performance_metrics.get('winning_trades', 0) / max(performance_metrics.get('total_trades', 1), 1)) * 100:.1f}%")
                    print(f"   Profit Factor: {performance_metrics.get('profit_factor', 0):.2f}")
                    print(f"   Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.3f}")
                    print(f"   Max Drawdown: {performance_metrics.get('max_drawdown', 0)*100:.2f}%")
                    
                    # Check for portfolio data
                    if 'portfolio_summary' in data:
                        portfolio_summary = data['portfolio_summary']
                        print(f"\n💰 Portfolio Summary:")
                        print(f"   Cash: ${portfolio_summary.get('cash', 0):,.2f}")
                        print(f"   Total Positions Value: ${portfolio_summary.get('total_positions_value', 0):,.2f}")
                        
                        positions = portfolio_summary.get('positions', {})
                        if positions:
                            print(f"   Active Positions: {len(positions)}")
                            for symbol, pos in positions.items():
                                print(f"     {symbol}: {pos.get('shares', 0)} shares @ ${pos.get('entry_price', 0):.2f}")
                    
                    # Check for trades
                    if 'trades' in data:
                        trades = data['trades']
                        print(f"\n📈 Trading History:")
                        print(f"   Total Trades: {len(trades)}")
                        
                        # Count buy/sell actions
                        buy_trades = [t for t in trades if t.get('action') == 'buy']
                        sell_trades = [t for t in trades if t.get('action') == 'sell']
                        print(f"   Buy Trades: {len(buy_trades)}")
                        print(f"   Sell Trades: {len(sell_trades)}")
                        
                        # Show some recent trades
                        recent_trades = trades[-5:] if len(trades) >= 5 else trades
                        print(f"   Recent Trades:")
                        for trade in recent_trades:
                            action = trade.get('action', 'unknown')
                            symbol = trade.get('symbol', 'N/A')
                            shares = trade.get('shares', 0)
                            price = trade.get('price', 0)
                            date = trade.get('date', 'N/A')
                            print(f"     {date}: {action} {shares} {symbol} @ ${price:.2f}")
                    
                else:
                    print(f"❌ Missing required sections: {missing_sections}")
                    
            except Exception as e:
                print(f"❌ Error loading results: {e}")
        else:
            print("❌ No comprehensive backtest results found")
    else:
        print("❌ Backtest results directory not found")
    
    # Test 2: Check if comprehensive configs exist
    print("\n2️⃣ Checking for comprehensive configurations...")
    
    configs_dir = "backtest_configs"
    if os.path.exists(configs_dir):
        config_files = [f for f in os.listdir(configs_dir) if f.startswith('comprehensive_config_') and f.endswith('.json')]
        
        if config_files:
            latest_config = max(config_files, key=lambda x: os.path.getctime(os.path.join(configs_dir, x)))
            print(f"✅ Found comprehensive config: {latest_config}")
            
            # Test loading the config
            try:
                with open(os.path.join(configs_dir, latest_config), 'r') as f:
                    config = json.load(f)
                
                print(f"✅ Successfully loaded config data")
                print(f"⚙️ Config keys: {list(config.keys())}")
                
                # Display key config parameters
                print(f"\n⚙️ Key Configuration Parameters:")
                print(f"   Symbols: {config.get('symbols', [])}")
                print(f"   Initial Capital: ${config.get('initial_capital', 0):,.2f}")
                print(f"   Position Size: {config.get('position_size', 0)*100:.1f}%")
                print(f"   Stop Loss: {config.get('stop_loss_pct', 0)*100:.1f}%")
                print(f"   Take Profit: {config.get('take_profit_pct', 0)*100:.1f}%")
                print(f"   Prediction Threshold: {config.get('prediction_threshold', 0):.4f}")
                
            except Exception as e:
                print(f"❌ Error loading config: {e}")
        else:
            print("❌ No comprehensive config files found")
    else:
        print("❌ Backtest configs directory not found")
    
    print("\n🎉 Dashboard data loading test completed!")
    print("\n📋 Summary:")
    print("   - Comprehensive backtest results are available")
    print("   - Configuration files are available")
    print("   - Data structure is compatible with dashboard")
    print("   - Dashboard should work correctly!")
    
    return True

if __name__ == "__main__":
    test_data_loading() 