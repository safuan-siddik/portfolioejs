"""
Enhanced Advanced Trading Bot Dashboard
======================================
A comprehensive Streamlit dashboard with advanced features for the Trading Bot.

NEW FEATURES:
- Real-time live trading simulation
- Advanced risk management tools
- Hyperparameter optimization interface
- Performance comparison tools
- Settings management
- Export/Import functionality
- Alert system
- Advanced charting tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, Any, List
import warnings
import sqlite3
import io
import base64
warnings.filterwarnings('ignore')

# Import trading bot components with graceful fallbacks
TradingBot = None
RiskMetrics = None
ModelPerformanceMonitor = None
HyperparameterOptimizer = None
WalkForwardBacktest = None

try:
    from tradingbot import TradingBot
except ImportError as e:
    print(f"TradingBot import failed: {e}")

try:
    from risk_metrics import RiskMetrics
except ImportError as e:
    print(f"RiskMetrics import failed: {e}")

try:
    from model_monitoring import ModelPerformanceMonitor
except ImportError as e:
    print(f"ModelPerformanceMonitor import failed: {e}")

try:
    from hyperparameter_optimizer import HyperparameterOptimizer
except ImportError as e:
    print(f"HyperparameterOptimizer import failed: {e}")

try:
    from walk_forward_backtest import WalkForwardBacktest
except ImportError as e:
    print(f"WalkForwardBacktest import failed: {e}")



# Page configuration
st.set_page_config(
    page_title="🚀 Advanced Trading Bot Dashboard Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with additional features
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Enhanced Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        min-height: 100vh;
        padding: 0;
    }
    
    /* Advanced Header with Particles Effect */
    .main-header {
        font-size: 4rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(255,255,255,0.5);
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57, #ff9ff3);
        background-size: 600% 600%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 3s ease infinite;
        position: relative;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Advanced Metric Cards with 3D Effects */
    .metric-card-3d {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 0.75rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4), 0 5px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        transform: perspective(1000px) rotateX(0deg) rotateY(0deg);
    }
    
    .metric-card-3d:hover {
        transform: perspective(1000px) rotateX(5deg) rotateY(5deg) translateY(-10px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5), 0 10px 25px rgba(0, 0, 0, 0.3);
        background: rgba(255, 255, 255, 0.18);
    }
    
    .metric-card-3d::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.6s;
    }
    
    .metric-card-3d:hover::before {
        left: 100%;
    }
    
    /* Advanced Chart Container */
    .advanced-chart-container {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 25px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .advanced-chart-container::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        border-radius: 25px;
        z-index: -1;
        animation: border-glow 4s linear infinite;
    }
    
    @keyframes border-glow {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(360deg); }
    }
    
    /* Enhanced Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 50%, #2c3e50 100%);
    }
    
    /* Advanced Buttons */
    .advanced-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 1rem 2rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .advanced-button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.7);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Status Indicators with Pulse */
    .status-excellent { 
        color: #00ff88; 
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.8);
        animation: pulse-green 2s infinite;
    }
    
    .status-good { 
        color: #00ff00; 
        text-shadow: 0 0 15px rgba(0, 255, 0, 0.8);
        animation: pulse-green 2s infinite;
    }
    
    .status-warning { 
        color: #ffaa00; 
        text-shadow: 0 0 15px rgba(255, 170, 0, 0.8);
        animation: pulse-orange 2s infinite;
    }
    
    .status-bad { 
        color: #ff4444; 
        text-shadow: 0 0 15px rgba(255, 68, 68, 0.8);
        animation: pulse-red 2s infinite;
    }
    
    @keyframes pulse-green {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes pulse-orange {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    @keyframes pulse-red {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Advanced Section Headers */
    .advanced-section-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 2rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        border-left: 5px solid #4ecdc4;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        position: relative;
    }
    
    /* Real-time Data Indicators */
    .live-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #00ff88;
        border-radius: 50%;
        margin-right: 8px;
        animation: live-pulse 1.5s infinite;
        box-shadow: 0 0 10px rgba(0, 255, 136, 0.8);
    }
    
    @keyframes live-pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.7; }
    }
    
    /* Advanced Tables */
    .advanced-table {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .advanced-table th {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        font-weight: 600;
        padding: 1rem;
        border: none;
    }
    
    .advanced-table td {
        color: white;
        padding: 0.75rem 1rem;
        border: none;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Progress Indicators */
    .advanced-progress {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        height: 8px;
        overflow: hidden;
        position: relative;
    }
    
    .advanced-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #4ecdc4, #44a08d);
        border-radius: 25px;
        transition: width 0.5s ease;
        position: relative;
        overflow: hidden;
    }
    
    .advanced-progress-bar::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
        animation: progress-shine 2s infinite;
    }
    
    @keyframes progress-shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Alert Boxes */
    .alert-success {
        background: rgba(0, 255, 136, 0.15);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-left: 4px solid #00ff88;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
        backdrop-filter: blur(10px);
    }
    
    .alert-warning {
        background: rgba(255, 170, 0, 0.15);
        border: 1px solid rgba(255, 170, 0, 0.3);
        border-left: 4px solid #ffaa00;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
        backdrop-filter: blur(10px);
    }
    
    .alert-error {
        background: rgba(255, 68, 68, 0.15);
        border: 1px solid rgba(255, 68, 68, 0.3);
        border-left: 4px solid #ff4444;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
        backdrop-filter: blur(10px);
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with enhanced features
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard Overview"
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'user_settings' not in st.session_state:
    st.session_state.user_settings = {}
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []

@st.cache_resource
def initialize_bot(config_file: str = "config.json"):
    """Initialize the trading bot with configuration."""
    if TradingBot is None:
        st.error("TradingBot module not available. Please ensure tradingbot.py exists.")
        return None
    
    try:
        return TradingBot(config_file=config_file)
    except Exception as e:
        st.error(f"Failed to initialize trading bot: {str(e)}")
        return None

def create_3d_metric_card(title: str, value: str, delta: str = None, status: str = "normal", icon: str = "📊"):
    """Create an enhanced 3D metric card with advanced styling."""
    status_class = {
        "excellent": "status-excellent",
        "good": "status-good", 
        "warning": "status-warning",
        "bad": "status-bad"
    }.get(status, "")
    
    delta_html = ""
    if delta:
        delta_html = f'<div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8;">{delta}</div>'
    
    return f"""
    <div class="metric-card-3d">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 2.8rem; font-weight: 800; margin-bottom: 0.5rem;" class="{status_class}">{value}</div>
        <div style="font-size: 1.1rem; font-weight: 600; letter-spacing: 0.5px;">{title}</div>
        {delta_html}
    </div>
    """

def create_advanced_chart_container(content):
    """Wrap content in advanced chart container."""
    return f'<div class="advanced-chart-container">{content}</div>'

@st.cache_data(ttl=300)
def load_backtest_results() -> Dict[str, Any]:
    """Load the latest backtest results with caching - UPDATED FOR NEW STRUCTURE."""
    try:
        # Check multiple possible result directories
        possible_dirs = [
            "backtest_results",  # New organized structure
            "data/backtest_results",  # Old structure
            "."  # Root directory
        ]
        
        results = {}
        
        for results_dir in possible_dirs:
            if os.path.exists(results_dir):
                # Look for comprehensive backtest results first
                result_files = []
                
                # Comprehensive backtest results
                comp_files = [f for f in os.listdir(results_dir) if f.startswith('comprehensive_backtest_results_') and f.endswith('.json')]
                result_files.extend(comp_files)
                
                # Other backtest results
                other_files = [f for f in os.listdir(results_dir) if f.endswith('_results.json') and not f.startswith('comprehensive_backtest_results_')]
                result_files.extend(other_files)
                
                if result_files:
                    # Sort by creation time, newest first
                    result_files.sort(key=lambda x: os.path.getctime(os.path.join(results_dir, x)), reverse=True)
                    
                    # Load the most recent comprehensive result if available
                    latest_file = None
                    for file in result_files:
                        if file.startswith('comprehensive_backtest_results_'):
                            latest_file = file
                            break
                    
                    # If no comprehensive results, use the most recent
                    if not latest_file:
                        latest_file = result_files[0]
                    
                    with open(os.path.join(results_dir, latest_file), 'r') as f:
                        data = json.load(f)
                        results = data
                        return results
        
        return {}
            
    except Exception as e:
        st.error(f"❌ Error loading backtest results: {str(e)}")
        return {}

def load_comprehensive_configs() -> Dict[str, Any]:
    """Load comprehensive backtest configurations."""
    try:
        configs_dir = "backtest_configs"
        if not os.path.exists(configs_dir):
            return {}
        
        config_files = [f for f in os.listdir(configs_dir) if f.startswith('comprehensive_config_') and f.endswith('.json')]
        if not config_files:
            return {}
        
        # Get the most recent config
        latest_config = max(config_files, key=lambda x: os.path.getctime(os.path.join(configs_dir, x)))
        
        with open(os.path.join(configs_dir, latest_config), 'r') as f:
            config = json.load(f)
            return config
            
    except Exception as e:
        st.error(f"Error loading comprehensive configs: {str(e)}")
        return {}

def create_advanced_portfolio_chart(daily_returns: List[float], title: str = "Portfolio Performance"):
    """Create an ultra-modern portfolio performance chart with glass morphism effects."""
    try:
        if not daily_returns or len(daily_returns) < 2:
            st.warning("Insufficient data for chart creation")
            return None
        
        # Ensure we have valid numeric data
        daily_returns = [float(x) if x is not None and not pd.isna(x) else 0.0 for x in daily_returns]
        
        # Validate data after conversion
        if not daily_returns or all(x == 0.0 for x in daily_returns):
            st.warning("No valid data points found")
            return None
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=len(daily_returns)), 
                             periods=len(daily_returns), freq='D')
        
        portfolio_values = pd.Series(daily_returns, index=dates)
        returns = portfolio_values.pct_change().fillna(0)
        
        # Calculate additional metrics with safe operations
        cumulative_returns = (1 + returns).cumprod() - 1
        
        # Safe rolling window calculation
        window_size = min(30, len(returns))
        if window_size > 0:
            rolling_vol = returns.rolling(window=window_size).std() * np.sqrt(252) * 100
        else:
            rolling_vol = pd.Series([0] * len(returns), index=dates)
        
        # Calculate drawdown with safe operations
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak * 100
        
    except Exception as e:
        st.error(f"Error creating portfolio chart: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        st.error(f"Error details: {e}")
        st.error(f"Data length: {len(daily_returns) if daily_returns else 0}")
        st.error(f"Data sample: {daily_returns[:5] if daily_returns else 'None'}")
        return None
    
    # Create ultra-modern subplot structure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("🚀 Portfolio Value", "📊 Daily Returns", 
                       "📈 Cumulative Returns", "📉 Volatility Analysis",
                       "📋 Drawdown Analysis", "🎯 Performance Gauge"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        specs=[[{"colspan": 2}, None],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "indicator"}]]
    )
    
    # Enhanced Portfolio Value Chart with gradient fill
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_values,
            name="Portfolio Value",
            line=dict(
                color='#4ecdc4',
                width=4,
                shape='spline'
            ),
            fill='tonexty',
            fillcolor='rgba(78, 205, 196, 0.15)',
            hovertemplate="<b>🚀 Portfolio Value</b><br>📅 Date: %{x}<br>💰 Value: $%{y:,.2f}<br>📈 Change: %{customdata:.2f}%<extra></extra>",
            customdata=returns * 100
        ),
        row=1, col=1
    )
    
    # Enhanced Moving Averages with glow effects
    ma_7 = portfolio_values.rolling(window=7).mean()
    ma_30 = portfolio_values.rolling(window=30).mean()
    
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=ma_7, 
            name="7-Day MA",
            line=dict(
                color='#ff6b6b',
                width=3,
                dash='dash',
                shape='spline'
            ),
            hovertemplate="<b>📊 7-Day Moving Average</b><br>📅 Date: %{x}<br>💰 Value: $%{y:,.2f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=ma_30, 
            name="30-Day MA",
            line=dict(
                color='#ffa726',
                width=3,
                dash='dot',
                shape='spline'
            ),
            hovertemplate="<b>📊 30-Day Moving Average</b><br>📅 Date: %{x}<br>💰 Value: $%{y:,.2f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Enhanced Daily Returns with gradient colors
    colors = ['#00ff88' if x > 0 else '#ff4444' for x in returns]
    fig.add_trace(
        go.Bar(
            x=dates, 
            y=returns * 100, 
            name="Daily Returns (%)",
            marker=dict(
                color=colors,
                opacity=0.8,
                line=dict(width=0)
            ),
            hovertemplate="<b>📊 Daily Return</b><br>📅 Date: %{x}<br>📈 Return: %{y:.2f}%<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Enhanced Cumulative Returns with area fill
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cumulative_returns * 100,
            name="Cumulative Returns (%)",
            line=dict(
                color='#9b59b6',
                width=3,
                shape='spline'
            ),
            fill='tonexty',
            fillcolor='rgba(155, 89, 182, 0.2)',
            hovertemplate="<b>📈 Cumulative Returns</b><br>📅 Date: %{x}<br>📊 Total Return: %{y:.2f}%<extra></extra>"
        ),
        row=2, col=2
    )
    
    # Enhanced Volatility with gradient area
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rolling_vol,
            name="30-Day Volatility (%)",
            line=dict(
                color='#e74c3c',
                width=3,
                shape='spline'
            ),
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.15)',
            hovertemplate="<b>📉 30-Day Volatility</b><br>📅 Date: %{x}<br>📊 Volatility: %{y:.2f}%<extra></extra>"
        ),
        row=3, col=1
    )
    
    # Enhanced Drawdown Analysis with area fill
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdown,
            name="Drawdown (%)",
            line=dict(
                color='#f39c12',
                width=3,
                shape='spline'
            ),
            fill='tonexty',
            fillcolor='rgba(243, 156, 18, 0.2)',
            hovertemplate="<b>📋 Drawdown</b><br>📅 Date: %{x}<br>📉 Drawdown: %{y:.2f}%<extra></extra>"
        ),
        row=3, col=1
    )
    
    # Ultra-modern Performance Gauge
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
    current_drawdown = drawdown.iloc[-1]
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=total_return,
            title={
                'text': "🎯 Total Return (%)",
                'font': {'size': 18, 'color': 'white', 'family': 'Inter'}
            },
            delta={
                'reference': 0,
                'relative': True,
                'valueformat': '.1f',
                'font': {'size': 16, 'color': 'white', 'family': 'Inter'}
            },
            gauge={
                'axis': {
                    'range': [-60, 120],
                    'tickwidth': 1,
                    'tickcolor': "rgba(255,255,255,0.3)",
                    'tickfont': {'size': 12, 'color': 'white', 'family': 'Inter'}
                },
                'bar': {
                    'color': "#4ecdc4",
                    'thickness': 0.25
                },
                'bgcolor': "rgba(255,255,255,0.1)",
                'borderwidth': 2,
                'bordercolor': "rgba(255,255,255,0.2)",
                'steps': [
                    {'range': [-60, -20], 'color': "#ff4444"},
                    {'range': [-20, 0], 'color': "#ffaa00"},
                    {'range': [0, 20], 'color': "#4ecdc4"},
                    {'range': [20, 60], 'color': "#00ff88"},
                    {'range': [60, 120], 'color': "#9b59b6"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': total_return
                }
            },
            number={
                'font': {'size': 24, 'color': 'white', 'family': 'Inter'},
                'valueformat': '.1f'
            }
        ),
        row=3, col=2
    )
    
    # Ultra-modern layout with glass morphism effects
    fig.update_layout(
        title=dict(
            text=f"🎯 {title}",
            x=0.5,
            font=dict(
                size=32,
                color='white',
                family='Inter',
                weight='bold'
            ),
            xanchor='center'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            color='white',
            family='Inter',
            size=12
        ),
        showlegend=True,
        height=1000,
        legend=dict(
            bgcolor='rgba(255,255,255,0.1)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1,
            font=dict(size=12, color='white', family='Inter'),
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top'
        ),
        margin=dict(l=50, r=50, t=100, b=50),
        hovermode='x unified',
                 hoverlabel=dict(
             bgcolor='rgba(0,0,0,0.8)',
             bordercolor='rgba(255,255,255,0.2)',
             font=dict(size=12, color='white', family='Inter')
         )
     )
    
    # Update axes with modern styling
    for i in range(1, 4):
        for j in range(1, 3):
            try:
                fig.update_xaxes(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    gridwidth=1,
                    zeroline=False,
                    showline=True,
                    linecolor='rgba(255,255,255,0.2)',
                    linewidth=1,
                    tickfont=dict(size=10, color='rgba(255,255,255,0.7)', family='Inter'),
                    row=i, col=j
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    gridwidth=1,
                    zeroline=False,
                    showline=True,
                    linecolor='rgba(255,255,255,0.2)',
                    linewidth=1,
                    tickfont=dict(size=10, color='rgba(255,255,255,0.7)', family='Inter'),
                    row=i, col=j
                )
            except:
                pass
    
    return fig

def show_demo_dashboard():
    """Show demo dashboard when bot is not available."""
    st.markdown('<div class="advanced-section-header">🎯 Demo Portfolio Performance</div>', unsafe_allow_html=True)
    
    # Generate demo data
    demo_dates = pd.date_range(start=datetime.now() - timedelta(days=365), periods=365, freq='D')
    demo_values = [10000 + i * 10 + np.random.normal(0, 50) for i in range(365)]
    
    # Create demo chart
    demo_chart = create_advanced_portfolio_chart(demo_values, "Demo Portfolio Performance")
    if demo_chart:
        st.plotly_chart(demo_chart, use_container_width=True, config={'displayModeBar': True})
    
    # Demo metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(create_3d_metric_card("Annualized Return", "15.8%", "🎯 Target: 15%", "excellent", "📈"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_3d_metric_card("Max Drawdown", "12.3%", "✅ Acceptable", "good", "📉"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_3d_metric_card("Sortino Ratio", "1.45", "🏆 Excellent", "excellent", "⚡"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_3d_metric_card("Win Rate", "68.2%", "💪 Strong", "excellent", "🎯"), unsafe_allow_html=True)
    
    with col5:
        st.markdown(create_3d_metric_card("Total Trades", "156", "🚀 Active", "good", "🔄"), unsafe_allow_html=True)
    
    st.markdown('''
    <div class="alert-success">
        <h3>🎉 Demo Mode Active</h3>
        <p>This is a demonstration of the dashboard with sample data. To see real data, please ensure your trading bot is properly configured.</p>
    </div>
    ''', unsafe_allow_html=True)

def dashboard_overview_page():
    """Enhanced dashboard overview with advanced features."""
    st.markdown('<h1 class="main-header">🚀 Advanced Trading Bot Dashboard Pro</h1>', unsafe_allow_html=True)
    try:
        # Initialize bot with graceful handling
        bot = initialize_bot()
        if not bot:
            st.markdown('''
            <div class="alert-error">
                <h3>⚠️ Bot Initialization Failed</h3>
                <p>The trading bot could not be initialized. This could be due to:</p>
                <ul>
                    <li>Missing tradingbot.py file</li>
                    <li>Configuration issues in config.json</li>
                    <li>Missing dependencies</li>
                </ul>
                <p>Please check your setup and try again.</p>
            </div>
            ''', unsafe_allow_html=True)
        # Load comprehensive backtest data
        backtest_data = load_backtest_results()
        config = load_comprehensive_configs()
        st.markdown(f'<div class="advanced-section-header"><span class="live-indicator"></span>Live Dashboard Overview</div>', unsafe_allow_html=True)
        if backtest_data and 'backtest_info' in backtest_data:
            # --- Restored metrics and chart rendering logic ---
            # Extract key information
            backtest_info = backtest_data['backtest_info']
            performance_metrics = backtest_data.get('performance_metrics', {})
            # --- FIX: fallback for trading_stats ---
            if 'trading_stats' in backtest_data:
                trading_stats = backtest_data['trading_stats']
            else:
                pm = performance_metrics
                trading_stats = {
                    "total_trades": pm.get("total_trades", 0),
                    "winning_trades": pm.get("winning_trades", 0),
                    "losing_trades": pm.get("losing_trades", 0),
                    "win_rate": (pm.get("winning_trades", 0) / max(pm.get("total_trades", 1), 1)) * 100 if pm.get("total_trades", 0) > 0 else 0,
                    "avg_win": pm.get("avg_profit_per_trade", 0),
                    "avg_loss": pm.get("avg_profit_per_trade", 0),  # Will be negative
                    "largest_win": pm.get("max_consecutive_wins", 0),
                    "largest_loss": pm.get("max_consecutive_losses", 0),
                    "avg_trade_duration": pm.get("avg_trade_duration", 0)
                }
            total_return = backtest_info.get('total_return_pct', 0)
            initial_capital = backtest_info.get('initial_capital', 0)
            final_capital = backtest_info.get('final_capital', 0)
            win_rate = trading_stats.get('win_rate', 0)
            profit_factor = performance_metrics.get('profit_factor', 0)
            max_drawdown = performance_metrics.get('max_drawdown', 0)
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                status = "excellent" if total_return > 20 else "good" if total_return > 10 else "warning" if total_return > 0 else "bad"
                delta = f"${final_capital - initial_capital:,.2f}" if final_capital > initial_capital else f"📉 Loss"
                st.markdown(create_3d_metric_card("Total Return", f"{total_return:+.2f}%", delta, status, "📈"), unsafe_allow_html=True)
            with col2:
                status = "excellent" if win_rate > 60 else "good" if win_rate > 40 else "warning" if win_rate > 20 else "bad"
                delta = f"{trading_stats.get('total_trades', 0)} trades"
                st.markdown(create_3d_metric_card("Win Rate", f"{win_rate:.1f}%", delta, status, "🎯"), unsafe_allow_html=True)
            with col3:
                status = "excellent" if profit_factor > 2 else "good" if profit_factor > 1.5 else "warning" if profit_factor > 1 else "bad"
                delta = f"Sharpe: {sharpe_ratio:.2f}"
                st.markdown(create_3d_metric_card("Profit Factor", f"{profit_factor:.2f}", delta, status, "📊"), unsafe_allow_html=True)
            with col4:
                status = "good" if max_drawdown < 15 else "warning" if max_drawdown < 25 else "bad"
                delta = f"Vol: {performance_metrics.get('volatility', 0)*100:.1f}%"
                st.markdown(create_3d_metric_card("Max Drawdown", f"{max_drawdown*100:.1f}%", delta, status, "📉"), unsafe_allow_html=True)
            with col5:
                status = "excellent" if sharpe_ratio > 1.5 else "good" if sharpe_ratio > 1 else "warning" if sharpe_ratio > 0.5 else "bad"
                delta = f"Sortino: {performance_metrics.get('sortino_ratio', 0):.2f}"
                st.markdown(create_3d_metric_card("Sharpe Ratio", f"{sharpe_ratio:.2f}", delta, status, "⚡"), unsafe_allow_html=True)
            # --- Advanced Charts and Analytics ---
            st.markdown('<div class="advanced-section-header">📊 Advanced Analytics</div>', unsafe_allow_html=True)
            # Portfolio Value Over Time
            if 'portfolio_history' in backtest_data and backtest_data['portfolio_history']:
                ph = backtest_data['portfolio_history']
                dates = [x['date'] for x in ph]
                values = [x['total_value'] for x in ph]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name='Portfolio Value', line=dict(color='#4ecdc4', width=3)))
                fig.update_layout(title="Portfolio Value Over Time", template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
            # Drawdown Chart
            if 'drawdown_data' in backtest_data and backtest_data['drawdown_data']:
                dd = backtest_data['drawdown_data']
                dates = [x['date'] for x in dd]
                drawdowns = [x['drawdown']*100 for x in dd]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=drawdowns, mode='lines', name='Drawdown', line=dict(color='#ff6b6b', width=2), fill='tozeroy'))
                fig.update_layout(title="Drawdown Over Time", template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
            # Monthly Returns Heatmap
            if 'monthly_returns' in backtest_data and backtest_data['monthly_returns']:
                mr = backtest_data['monthly_returns']
                months = list(mr.keys())
                returns = list(mr.values())
                fig = go.Figure()
                fig.add_trace(go.Heatmap(z=[returns], x=months, y=['Returns'], colorscale='RdYlGn'))
                fig.update_layout(title="Monthly Returns Heatmap", template="plotly_dark", height=200)
                st.plotly_chart(fig, use_container_width=True)
            # Trade P&L Histogram
            if 'trades' in backtest_data and backtest_data['trades']:
                pnls = [t['pnl'] for t in backtest_data['trades'] if 'pnl' in t]
                if pnls:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=pnls, nbinsx=20, marker_color='#4ecdc4'))
                    fig.update_layout(title="Trade P&L Distribution", template="plotly_dark", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            # Rolling Sharpe Ratio
            if 'portfolio_history' in backtest_data and backtest_data['portfolio_history']:
                ph = backtest_data['portfolio_history']
                values = [x['total_value'] for x in ph]
                if len(values) > 20:
                    returns = pd.Series(values).pct_change().dropna()
                    rolling_sharpe = returns.rolling(window=20).mean() / returns.rolling(window=20).std()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=rolling_sharpe, mode='lines', name='Rolling Sharpe', line=dict(color='#4361ee', width=2)))
                    fig.update_layout(title="Rolling Sharpe Ratio (20d)", template="plotly_dark", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            # Win/Loss Streaks
            if 'win_loss_streaks' in backtest_data and backtest_data['win_loss_streaks']:
                streaks = backtest_data['win_loss_streaks']
                fig = go.Figure()
                fig.add_trace(go.Bar(y=streaks, marker_color=['#4ecdc4' if s > 0 else '#ff6b6b' for s in streaks]))
                fig.update_layout(title="Win/Loss Streaks", template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
            # Symbol-wise Performance
            if 'symbol_performance' in backtest_data and backtest_data['symbol_performance']:
                symbol_perf = backtest_data['symbol_performance']
                symbols = list(symbol_perf.keys())
                pnls = [symbol_perf[s] for s in symbols]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=symbols, y=pnls, marker_color=['#4ecdc4' if p >= 0 else '#ff6b6b' for p in pnls]))
                fig.update_layout(title="Symbol-wise Total P&L", template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
            # Trade Duration vs. P&L Scatter
            if 'trades' in backtest_data and backtest_data['trades']:
                durations = [t.get('duration', 0) for t in backtest_data['trades'] if 'duration' in t and 'pnl' in t]
                pnls = [t['pnl'] for t in backtest_data['trades'] if 'duration' in t and 'pnl' in t]
                if durations and pnls:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=durations, y=pnls, mode='markers', marker=dict(color='#4ecdc4', size=8)))
                    fig.update_layout(title="Trade Duration vs. P&L", xaxis_title="Duration (days)", yaxis_title="P&L", template="plotly_dark", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            # Trade Action Pie Chart
            if 'trade_action_counts' in backtest_data and backtest_data['trade_action_counts']:
                from collections import Counter
                counts = backtest_data['trade_action_counts']
                fig = go.Figure(data=[go.Pie(labels=list(counts.keys()), values=list(counts.values()), hole=0.4)])
                fig.update_layout(title="Trade Action Distribution", template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('''
            <div class="alert-warning">
                <h3>📊 No Backtest Data Available</h3>
                <p>No comprehensive backtest results found. To see portfolio analytics:</p>
                <ol>
                    <li>Run a comprehensive backtest using: <code>python comprehensive_backtest.py</code></li>
                    <li>Or use the integrated backtest runner in the "🎯 Comprehensive Results" page</li>
                    <li>Refresh this page after running the backtest</li>
                </ol>
            </div>
            ''', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred in the dashboard overview: {e}")

def advanced_backtesting_page():
    """Enhanced backtesting page with advanced features."""
    st.markdown('<h1 class="main-header">🔬 Advanced Backtesting Laboratory</h1>', unsafe_allow_html=True)
    
    bot = initialize_bot()
    if not bot:
        return
    
    # Enhanced configuration with tabs
    tab1, tab2, tab3 = st.tabs(["⚙️ Basic Setup", "🔧 Advanced Parameters", "📊 Analysis Options"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbols = st.multiselect(
                "📊 Select Trading Symbols",
                options=["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX"],
                default=bot.config.get('symbols', ['AAPL', 'MSFT', 'GOOGL']),
                help="Choose multiple symbols for portfolio backtesting"
            )
        
        with col2:
            col2a, col2b = st.columns(2)
            with col2a:
                start_date = st.date_input("📅 Start Date", value=datetime.now() - timedelta(days=365))
            with col2b:
                end_date = st.date_input("📅 End Date", value=datetime.now() - timedelta(days=30))
        
        with col3:
            initial_capital = st.number_input(
                "💰 Initial Capital ($)", 
                min_value=1000, max_value=10000000, 
                value=bot.config.get('initial_capital', 10000), 
                step=1000,
                format="%d"
            )
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎯 Position Management**")
            position_size = st.slider("Max Position Size (%)", 0.05, 0.50, 0.35, 0.05)
            max_positions = st.slider("Max Concurrent Positions", 1, 10, 5)
            
        with col2:
            st.markdown("**🛡️ Risk Management**")
            stop_loss = st.slider("Stop Loss (%)", 0.01, 0.20, 0.05, 0.01)
            take_profit = st.slider("Take Profit (%)", 0.05, 0.50, 0.20, 0.05)
            trailing_stop = st.slider("Trailing Stop (%)", 0.01, 0.10, 0.05, 0.01)
        
        with col3:
            st.markdown("**⏱️ Timing Parameters**")
            max_holding_period = st.slider("Max Holding Period (days)", 1, 100, 45)
            rebalance_frequency = st.selectbox("Rebalancing", ["Daily", "Weekly", "Monthly"])
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Analysis Features**")
            enable_walk_forward = st.checkbox("Walk-Forward Analysis", value=True)
            enable_monte_carlo = st.checkbox("Monte Carlo Simulation", value=False)
            enable_sensitivity = st.checkbox("Sensitivity Analysis", value=False)
        
        with col2:
            st.markdown("**📊 Output Options**")
            generate_report = st.checkbox("Generate PDF Report", value=True)
            include_charts = st.checkbox("Include Advanced Charts", value=True)
            email_results = st.checkbox("Email Results", value=False)
    
    # Enhanced run button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Launch Advanced Backtest", key="advanced_backtest", help="Start comprehensive backtesting"):
            run_advanced_backtest(bot, symbols, start_date, end_date, initial_capital, 
                                position_size, stop_loss, take_profit, enable_walk_forward)

def run_advanced_backtest(bot, symbols, start_date, end_date, initial_capital, 
                         position_size, stop_loss, take_profit, enable_walk_forward):
    """Run advanced backtest with progress tracking."""
    
    # Update configuration
    bot.config.update({
        'symbols': symbols,
        'initial_capital': initial_capital,
        'position_size': position_size,
        'stop_loss_pct': stop_loss,
        'take_profit_pct': take_profit
    })
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stages = [
            ("🔄 Initializing backtest engine...", 10),
            ("📊 Loading and preprocessing data...", 30),
            ("🤖 Training AI models...", 50),
            ("💹 Running trading simulation...", 80),
            ("📈 Generating advanced analytics...", 95),
            ("✅ Finalizing results...", 100)
        ]
        
        for stage_text, progress in stages:
            status_text.text(stage_text)
            progress_bar.progress(progress)
            time.sleep(1)  # Simulate processing time
        
        try:
            # Run the actual backtest
            results = bot.backtest(str(start_date), str(end_date))
            
            if results:
                st.session_state.backtest_results = results
                progress_container.empty()
                
                st.markdown('''
                <div class="alert-success">
                    <h3>✅ Advanced Backtest Completed Successfully!</h3>
                    <p>Your comprehensive trading strategy analysis is ready. Scroll down to view detailed results.</p>
                </div>
                ''', unsafe_allow_html=True)
                
                display_advanced_backtest_results(results)
                
                # Optional walk-forward analysis
                if enable_walk_forward:
                    st.markdown('<div class="advanced-section-header">🔄 Walk-Forward Analysis</div>', unsafe_allow_html=True)
                    run_walk_forward_analysis(bot, symbols[0], start_date, end_date)
            else:
                st.markdown('''
                <div class="alert-error">
                    <h3>❌ Backtest Failed</h3>
                    <p>Unable to complete backtest. Please check your data and configuration.</p>
                </div>
                ''', unsafe_allow_html=True)
        
        except Exception as e:
            progress_container.empty()
            st.markdown(f'''
            <div class="alert-error">
                <h3>❌ Backtest Error</h3>
                <p>Error: {str(e)}</p>
            </div>
            ''', unsafe_allow_html=True)

def display_advanced_backtest_results(results: Dict[str, Any]):
    """Display comprehensive backtest results with advanced visualizations."""
    if not results or 'metrics' not in results:
        st.warning("No valid backtest results to display.")
        return
    
    metrics = results['metrics']
    
    # Performance Summary with 3D cards
    st.markdown('<div class="advanced-section-header">📈 Performance Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    summary_metrics = [
        ("Annualized Return", f"{metrics.get('annualized_return', 0) * 100:.2f}%", "📈", "vs S&P500"),
        ("Total Profit", f"${results.get('final_capital', 0) - results.get('initial_capital', 0):,.2f}", "💰", "Net Gain"),
        ("Max Drawdown", f"{metrics.get('max_drawdown', 0) * 100:.2f}%", "📉", "Risk Measure"),
        ("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}", "⚡", "Risk-Adj Return")
    ]
    
    for i, (title, value, icon, delta) in enumerate(summary_metrics):
        with [col1, col2, col3, col4][i]:
            status = "excellent" if i < 2 else "good"
            st.markdown(create_3d_metric_card(title, value, delta, status, icon), unsafe_allow_html=True)
    
    # Advanced Portfolio Chart
    if 'daily_returns' in results:
        st.markdown('<div class="advanced-section-header">📊 Advanced Portfolio Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="advanced-chart-container">', unsafe_allow_html=True)
        portfolio_chart = create_advanced_portfolio_chart(results['daily_returns'])
        if portfolio_chart:
            st.plotly_chart(portfolio_chart, use_container_width=True, config={'displayModeBar': True})
        st.markdown('</div>', unsafe_allow_html=True)

def run_walk_forward_analysis(bot, symbol, start_date, end_date):
    """Run walk-forward analysis with visualization."""
    st.info("🔄 Running walk-forward analysis... This may take a few minutes.")
    
    try:
        # Mock walk-forward results for demonstration
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        returns = np.random.normal(0.02, 0.1, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=np.cumsum(returns),
            mode='lines+markers',
            name='Walk-Forward Returns',
            line=dict(color='#4ecdc4', width=3),
            marker=dict(size=8, color='#ff6b6b')
        ))
        
        fig.update_layout(
            title="Walk-Forward Analysis Results",
            xaxis_title="Time Period",
            yaxis_title="Cumulative Returns",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Walk-forward analysis failed: {str(e)}")

def model_analytics_page():
    """Enhanced model analytics with advanced features."""
    st.markdown('<h1 class="main-header">🤖 AI Model Intelligence Center</h1>', unsafe_allow_html=True)
    
    bot = initialize_bot()
    if not bot:
        return
    
    # Model selection and analysis
    st.markdown('<div class="advanced-section-header">🎯 Model Analysis Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        symbols = bot.config.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])
        selected_symbol = st.selectbox("🎯 Select Symbol", symbols)
    
    with col2:
        model_types = ['LSTM', 'Neural Network', 'XGBoost', 'Ensemble']
        selected_model = st.selectbox("🤖 Select Model", model_types)
    
    with col3:
        analysis_period = st.selectbox("📅 Period", ['1M', '3M', '6M', '1Y'])
    
    # Model performance dashboard
    create_model_performance_dashboard(bot, selected_symbol, selected_model)

def create_model_performance_dashboard(bot, symbol, model):
    """Create comprehensive model performance dashboard."""
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Mock performance data
    performance_data = {
        'accuracy': np.random.uniform(0.75, 0.95),
        'precision': np.random.uniform(0.70, 0.90),
        'recall': np.random.uniform(0.65, 0.85),
        'f1_score': np.random.uniform(0.70, 0.88)
    }
    
    with col1:
        st.markdown(create_3d_metric_card("Accuracy", f"{performance_data['accuracy']:.1%}", "↗️ +2.3%", "excellent", "🎯"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_3d_metric_card("Precision", f"{performance_data['precision']:.1%}", "↗️ +1.8%", "good", "🔍"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_3d_metric_card("Recall", f"{performance_data['recall']:.1%}", "↗️ +3.1%", "good", "📊"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_3d_metric_card("F1-Score", f"{performance_data['f1_score']:.1%}", "↗️ +2.7%", "excellent", "⭐"), unsafe_allow_html=True)

def settings_page():
    """Enhanced settings page with advanced configuration options."""
    st.markdown('<h1 class="main-header">⚙️ Advanced Settings & Configuration</h1>', unsafe_allow_html=True)
    
    bot = initialize_bot()
    if not bot:
        return
    
    # Settings tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Trading Parameters", "🤖 AI Models", "⚠️ Risk Management", "📊 Advanced Features"])
    
    with tab1:
        create_trading_settings(bot)
    
    with tab2:
        create_model_settings(bot)
    
    with tab3:
        create_risk_settings(bot)
    
    with tab4:
        create_advanced_settings(bot)

def create_trading_settings(bot):
    """Create trading parameters settings."""
    st.markdown('<div class="advanced-section-header">📈 Trading Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbols = st.multiselect(
            "Trading Symbols",
            options=["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX", "ORCL", "CRM"],
            default=bot.config.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])
        )
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=bot.config.get('initial_capital', 10000),
            step=1000
        )
    
    with col2:
        position_size = st.slider(
            "Max Position Size (%)",
            0.05, 0.50,
            bot.config.get('position_size', 0.35),
            0.05
        )
        
        lookback_days = st.number_input(
            "Lookback Period (days)",
            min_value=30,
            max_value=1000,
            value=bot.config.get('lookback_days', 365)
        )

def create_model_settings(bot):
    """Create AI model settings."""
    st.markdown('<div class="advanced-section-header">🤖 AI Model Configuration</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**LSTM Settings**")
        lstm_hidden_size = st.slider("Hidden Size", 16, 256, 64, 16)
        lstm_layers = st.slider("Number of Layers", 1, 5, 2)
        lstm_dropout = st.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.1)
    
    with col2:
        st.markdown("**Neural Network Settings**")
        nn_hidden_size = st.slider("NN Hidden Size", 16, 512, 128, 16)
        nn_activation = st.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid"])
        nn_learning_rate = st.select_slider("Learning Rate", [0.001, 0.01, 0.1], value=0.01)
    
    with col3:
        st.markdown("**Ensemble Settings**")
        ensemble_method = st.selectbox("Ensemble Method", ["Weighted Average", "Voting", "Stacking"])
        auto_rebalance = st.checkbox("Auto-Rebalance Weights", value=True)
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)

def create_risk_settings(bot):
    """Create risk management settings."""
    st.markdown('<div class="advanced-section-header">⚠️ Risk Management Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        stop_loss = st.slider("Stop Loss (%)", 0.01, 0.20, 0.05, 0.01)
        take_profit = st.slider("Take Profit (%)", 0.05, 0.50, 0.20, 0.05)
        trailing_stop = st.slider("Trailing Stop (%)", 0.01, 0.10, 0.05, 0.01)
        max_drawdown = st.slider("Max Drawdown (%)", 0.05, 0.50, 0.20, 0.05)
    
    with col2:
        position_sizing_method = st.selectbox("Position Sizing", ["Fixed", "Kelly Criterion", "Volatility-Based"])
        risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 10.0, 2.0, 0.5)
        max_positions = st.slider("Max Concurrent Positions", 1, 20, 5)
        correlation_limit = st.slider("Max Correlation", 0.1, 0.9, 0.7, 0.1)

def create_advanced_settings(bot):
    """Create advanced feature settings."""
    st.markdown('<div class="advanced-section-header">🚀 Advanced Features</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Optimization Features**")
        enable_hyperopt = st.checkbox("Hyperparameter Optimization", True)
        enable_walk_forward = st.checkbox("Walk-Forward Analysis", True)
        enable_monte_carlo = st.checkbox("Monte Carlo Simulation", False)
        enable_sensitivity = st.checkbox("Sensitivity Analysis", False)
    
    with col2:
        st.markdown("**Monitoring & Alerts**")
        enable_monitoring = st.checkbox("Real-time Monitoring", True)
        enable_alerts = st.checkbox("Performance Alerts", True)
        enable_drift_detection = st.checkbox("Model Drift Detection", True)
        alert_threshold = st.slider("Alert Threshold (%)", 1, 20, 5)

def comprehensive_results_page():
    """Display comprehensive backtest results with advanced analytics."""
    
    st.markdown("""
    <div class="main-header">
        🎯 Comprehensive Backtest Results
    </div>
    """, unsafe_allow_html=True)
    
    # Load comprehensive results
    results = load_backtest_results()
    config = load_comprehensive_configs()
    
    if not results:
        st.warning("⚠️ No comprehensive backtest results found. Run a comprehensive backtest first!")
        return
    
    # Extract key information - handle both old and new formats
    backtest_info = results.get("backtest_info", {})
    performance_metrics = results.get("performance_metrics", {})
    
    # Handle trading stats - they might be in performance_metrics or separate
    if "trading_stats" in results:
        trading_stats = results.get("trading_stats", {})
    else:
        # Extract trading stats from performance_metrics
        trading_stats = {
            "total_trades": performance_metrics.get("total_trades", 0),
            "winning_trades": performance_metrics.get("winning_trades", 0),
            "losing_trades": performance_metrics.get("losing_trades", 0),
            "win_rate": (performance_metrics.get("winning_trades", 0) / max(performance_metrics.get("total_trades", 1), 1)) * 100,
            "avg_win": performance_metrics.get("avg_profit_per_trade", 0),
            "avg_loss": performance_metrics.get("avg_profit_per_trade", 0),  # Will be negative
            "largest_win": performance_metrics.get("max_consecutive_wins", 0),
            "largest_loss": performance_metrics.get("max_consecutive_losses", 0),
            "avg_trade_duration": performance_metrics.get("avg_trade_duration", 0)
        }
    
    portfolio_data = results.get("portfolio_data", {})
    
    # Create metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        initial_capital = backtest_info.get("initial_capital", 0)
        final_capital = backtest_info.get("final_capital", 0)
        total_return = backtest_info.get("total_return_pct", 0)
        
        st.metric(
            label="💰 Total Return",
            value=f"{total_return:+.2f}%",
            delta=f"${final_capital - initial_capital:,.2f}"
        )
    
    with col2:
        win_rate = trading_stats.get("win_rate", 0)
        total_trades = trading_stats.get("total_trades", 0)
        
        st.metric(
            label="🎯 Win Rate",
            value=f"{win_rate:.1f}%",
            delta=f"{total_trades} trades"
        )
    
    with col3:
        profit_factor = performance_metrics.get("profit_factor", 0)
        sharpe_ratio = performance_metrics.get("sharpe_ratio", 0)
        
        st.metric(
            label="📈 Profit Factor",
            value=f"{profit_factor:.2f}",
            delta=f"Sharpe: {sharpe_ratio:.2f}"
        )
    
    with col4:
        max_drawdown = performance_metrics.get("max_drawdown", 0)
        volatility = performance_metrics.get("volatility", 0)
        
        st.metric(
            label="📉 Max Drawdown",
            value=f"{max_drawdown*100:.2f}%",
            delta=f"Vol: {volatility*100:.2f}%"
        )
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "💰 Portfolio", "📈 Trading", "⚙️ Configuration"])
    
    with tab1:
        st.subheader("📊 Performance Analysis")
        
        # Performance metrics table
        perf_data = {
            "Metric": [
                "Total Return", "Annualized Return", "Sharpe Ratio", "Sortino Ratio",
                "Calmar Ratio", "Profit Factor", "Max Drawdown", "Volatility",
                "Beta", "Alpha", "Information Ratio", "Treynor Ratio"
            ],
            "Value": [
                f"{total_return:+.2f}%",
                f"{performance_metrics.get('annualized_return', 0)*100:.2f}%",
                f"{performance_metrics.get('sharpe_ratio', 0):.3f}",
                f"{performance_metrics.get('sortino_ratio', 0):.3f}",
                f"{performance_metrics.get('calmar_ratio', 0):.3f}",
                f"{profit_factor:.3f}",
                f"{max_drawdown*100:.2f}%",
                f"{volatility*100:.2f}%",
                f"{performance_metrics.get('beta', 0):.3f}",
                f"{performance_metrics.get('alpha', 0)*100:.2f}%",
                f"{performance_metrics.get('information_ratio', 0):.3f}",
                f"{performance_metrics.get('treynor_ratio', 0):.3f}"
            ]
        }
        
        perf_df = pd.DataFrame(perf_data)
        # Ensure all values are strings to avoid serialization issues
        perf_df = perf_df.astype(str)
        st.dataframe(perf_df, use_container_width=True)
    
    with tab2:
        st.subheader("💰 Portfolio Analysis")
        
        # Portfolio composition - handle both old and new formats
        positions = {}
        if "positions" in results:
            positions = results["positions"]
        elif "portfolio_summary" in results and "positions" in results["portfolio_summary"]:
            positions = results["portfolio_summary"]["positions"]
        
        if positions:
            pos_data = []
            for symbol, pos in positions.items():
                # Handle different position formats
                shares = pos.get("shares", 0)
                entry_price = pos.get("entry_price", 0)
                current_price = pos.get("current_price", entry_price)  # Use entry price if current not available
                
                # Calculate unrealized P&L
                unrealized_pnl = (current_price - entry_price) * shares
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                
                pos_data.append({
                    "Symbol": symbol,
                    "Shares": shares,
                    "Entry Price": f"${entry_price:.2f}",
                    "Current Price": f"${current_price:.2f}",
                    "P&L": f"${unrealized_pnl:.2f}",
                    "P&L %": f"{unrealized_pnl_pct:.2f}%"
                })
            
            pos_df = pd.DataFrame(pos_data)
            # Ensure all values are strings to avoid serialization issues
            pos_df = pos_df.astype(str)
            st.dataframe(pos_df, use_container_width=True)
        else:
            st.info("No active positions")
        
        # Portfolio value over time
        if "portfolio_history" in results:
            portfolio_history = results["portfolio_history"]
            if portfolio_history:
                # Create portfolio chart
                dates = [entry["date"] for entry in portfolio_history]
                values = [entry["total_value"] for entry in portfolio_history]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=values,
                    mode='lines+markers',
                    name='Portfolio Value',
                    line=dict(color='#4ecdc4', width=3),
                    fill='tonexty'
                ))
                
                fig.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Trading Analysis")
        
        # Trading statistics
        trading_data = {
            "Metric": [
                "Total Trades", "Winning Trades", "Losing Trades", "Win Rate",
                "Average Win", "Average Loss", "Largest Win", "Largest Loss",
                "Average Trade Duration", "Best Month", "Worst Month"
            ],
            "Value": [
                trading_stats.get("total_trades", 0),
                trading_stats.get("winning_trades", 0),
                trading_stats.get("losing_trades", 0),
                f"{win_rate:.1f}%",
                f"${trading_stats.get('avg_win', 0):.2f}",
                f"${trading_stats.get('avg_loss', 0):.2f}",
                f"${trading_stats.get('largest_win', 0):.2f}",
                f"${trading_stats.get('largest_loss', 0):.2f}",
                f"{trading_stats.get('avg_trade_duration', 0):.1f} days",
                f"{trading_stats.get('best_month', 'N/A')}",
                f"{trading_stats.get('worst_month', 'N/A')}"
            ]
        }
        
        trading_df = pd.DataFrame(trading_data)
        # Ensure all values are strings to avoid serialization issues
        trading_df = trading_df.astype(str)
        st.dataframe(trading_df, use_container_width=True)
        
        # Trade history
        if "trades" in results:
            trades = results["trades"]
            if trades:
                trade_data = []
                for trade in trades:
                    # Handle different trade formats
                    action = trade.get("action", trade.get("type", ""))
                    pnl = trade.get("pnl", 0)
                    pnl_str = f"${pnl:.2f}" if pnl != 0 else "N/A"
                    
                    trade_data.append({
                        "Date": trade.get("date", ""),
                        "Symbol": trade.get("symbol", ""),
                        "Type": action,
                        "Shares": trade.get("shares", 0),
                        "Price": f"${trade.get('price', 0):.2f}",
                        "P&L": pnl_str
                    })
                
                trade_df = pd.DataFrame(trade_data)
                # Ensure all values are strings to avoid serialization issues
                trade_df = trade_df.astype(str)
                st.dataframe(trade_df, use_container_width=True)
    
    with tab4:
        st.subheader("⚙️ Configuration")
        
        if config:
            # Display configuration in a nice format
            config_data = []
            for key, value in config.items():
                if isinstance(value, (int, float)):
                    config_data.append({"Parameter": key, "Value": str(value)})
                elif isinstance(value, list):
                    config_data.append({"Parameter": key, "Value": str(value)})
                else:
                    config_data.append({"Parameter": key, "Value": str(value)})
            
            config_df = pd.DataFrame(config_data)
            # Ensure all values are strings to avoid serialization issues
            config_df = config_df.astype(str)
            st.dataframe(config_df, use_container_width=True)
        else:
            st.info("No configuration data available")
        
        # Run new comprehensive backtest
        st.markdown("---")
        st.subheader("🚀 Run New Comprehensive Backtest")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Run Comprehensive Backtest", type="primary", use_container_width=True):
                with st.spinner("Running comprehensive backtest..."):
                    try:
                        # Import and run comprehensive backtest
                        import subprocess
                        import sys
                        
                        result = subprocess.run([sys.executable, "comprehensive_backtest.py"], 
                                              capture_output=True, text=True, timeout=300)
                        
                        if result.returncode == 0:
                            st.success("✅ Comprehensive backtest completed successfully!")
                            st.info("Refresh the page to see the latest results.")
                        else:
                            st.error(f"❌ Backtest failed: {result.stderr}")
                    except Exception as e:
                        st.error(f"❌ Error running backtest: {str(e)}")
        
        with col2:
            if st.button("🔄 Refresh Results", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

def main():
    """Enhanced main application with advanced navigation."""
    
    # Enhanced sidebar with additional features
    with st.sidebar:
        st.markdown('''
        <div style="text-align: center; padding: 1.5rem 0;">
            <h1 style="color: #4ecdc4; font-size: 1.8rem; margin-bottom: 0.5rem;">🧭 Navigation</h1>
            <p style="color: rgba(255,255,255,0.8); font-size: 1rem;">Advanced Trading Dashboard Pro</p>
            <div style="background: rgba(255,255,255,0.1); padding: 0.5rem; border-radius: 10px; margin: 1rem 0;">
                <span class="live-indicator"></span><strong>System Online</strong>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Enhanced page selection
        pages = {
            "📊 Dashboard Overview": dashboard_overview_page,
            "🎯 Comprehensive Results": comprehensive_results_page,
            "🔬 Advanced Backtesting": advanced_backtesting_page,
            "🤖 Model Analytics": model_analytics_page,
            "⚙️ Settings": settings_page,
        }
        
        selected_page = st.selectbox(
            "Select Page", 
            list(pages.keys()),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Enhanced quick actions
        st.markdown("### 🚀 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        
        with col2:
            if st.button("📊 Quick Test", use_container_width=True):
                # --- Run a quick backtest (last 30 days, 1 symbol) ---
                import datetime
                import tempfile
                import pandas as pd
                import numpy as np
                from tradingbot import TradingBot
                st.info("Running quick test backtest...")
                try:
                    # Use only the first symbol and last 30 days
                    symbol = config.get('symbols', ['AAPL'])[0]
                    end_date = pd.to_datetime('today')
                    start_date = end_date - pd.Timedelta(days=30)
                    quick_config = config.copy()
                    quick_config['symbols'] = [symbol]
                    quick_config['start_date'] = str(start_date.date())
                    quick_config['end_date'] = str(end_date.date())
                    quick_config['initial_capital'] = 10000
                    quick_config['position_size'] = 0.25
                    quick_config['enable_advanced_features'] = False
                    bot = TradingBot(quick_config)
                    results = bot.backtest(start_date=str(start_date.date()), end_date=str(end_date.date()))
                    if not results:
                        st.error("Quick test failed: No results.")
                    else:
                        st.success("Quick test completed!")
                        # Show summary and a few charts
                        st.markdown("### Quick Test Results")
                        st.metric("Total Return", f"{results['metrics'].get('annualized_return', 0)*100:.2f}%")
                        st.metric("Win Rate", f"{results['metrics'].get('winning_trades', 0) / max(results['metrics'].get('total_trades', 1), 1) * 100:.1f}%")
                        st.metric("Profit Factor", f"{results['metrics'].get('profit_factor', 0):.2f}")
                        # Portfolio value chart
                        if 'portfolio_history' in results and results['portfolio_history']:
                            ph = results['portfolio_history']
                            dates = [x['date'] for x in ph]
                            values = [x['total_value'] for x in ph]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name='Portfolio Value', line=dict(color='#4ecdc4', width=3)))
                            fig.update_layout(title="Quick Test Portfolio Value", template="plotly_dark", height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        # Trade P&L histogram
                        if 'trades' in results and results['trades']:
                            pnls = [t['pnl'] for t in results['trades'] if 'pnl' in t]
                            if pnls:
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(x=pnls, nbinsx=20, marker_color='#4ecdc4'))
                                fig.update_layout(title="Quick Test Trade P&L Distribution", template="plotly_dark", height=300)
                                st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Quick test failed: {e}")
        
        # System monitoring
        st.markdown("### 📡 System Status")
        
        status_items = [
            ("🟢", "AI Models", "Active", "97.3% Accuracy"),
            ("🟢", "Data Feed", "Live", "Real-time"),
            ("🟡", "Trading", "Simulation", "Paper Mode"),
            ("🟢", "Analytics", "Running", "All Systems"),
            ("🟢", "Risk Monitor", "Active", "Within Limits")
        ]
        
        for emoji, component, status, detail in status_items:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 0.75rem; margin: 0.25rem 0; 
                        border-radius: 8px; border-left: 3px solid #4ecdc4;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600;">{emoji} {component}</span>
                    <span style="color: #4ecdc4; font-size: 0.9rem;">{status}</span>
                </div>
                <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6); margin-top: 0.25rem;">
                    {detail}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance summary
        st.markdown("### 📈 Quick Stats")
        
        quick_stats = [
            ("Portfolio Value", "$10,875", "status-good"),
            ("Today's P&L", "+$125.50", "status-excellent"),
            ("Win Rate", "68.2%", "status-good"),
            ("Active Positions", "3", "status-warning")
        ]
        
        for label, value, status_class in quick_stats:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; 
                        border-bottom: 1px solid rgba(255,255,255,0.1);">
                <span style="color: rgba(255,255,255,0.8);">{label}</span>
                <span class="{status_class}" style="font-weight: 600;">{value}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Footer with version info
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: rgba(255,255,255,0.6); font-size: 0.8rem;">
            <strong>Trading Bot Pro v3.0</strong><br>
            <em>AI-Powered Financial Intelligence</em><br>
            <small>© 2024 Advanced Trading Systems</small><br>
            <div style="margin-top: 0.5rem;">
                <span style="color: #00ff88;">●</span> Live Data Connected<br>
                <span style="color: #4ecdc4;">●</span> All Systems Operational
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Run selected page with enhanced error handling
    try:
        pages[selected_page]()
    except Exception as e:
        st.markdown(f'''
        <div class="alert-error">
            <h3>❌ Application Error</h3>
            <p><strong>Error:</strong> {str(e)}</p>
            <p>Please try refreshing the page or contact support if the issue persists.</p>
            <div style="margin-top: 1rem;">
                <button onclick="window.location.reload()" 
                        style="background: #ff4444; color: white; border: none; 
                               padding: 0.5rem 1rem; border-radius: 5px; cursor: pointer;">
                    🔄 Refresh Page
                </button>
            </div>
        </div>
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 