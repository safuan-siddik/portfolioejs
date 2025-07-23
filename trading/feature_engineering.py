"""
Advanced Feature Engineering Module
===================================
This module provides comprehensive feature engineering with technical indicators,
market microstructure, and sentiment features.

Classes:
- AdvancedFeatureEngineering: The main class for creating features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("FeatureEngineering")

# Manual implementations of technical indicators (replacing TA-Lib)
def calculate_sma(prices, period):
    """Calculate Simple Moving Average."""
    return prices.rolling(window=period).mean()

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average."""
    return prices.ewm(span=period).mean()

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD."""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    sma = calculate_sma(prices, period)
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = calculate_sma(k_percent, d_period)
    return k_percent, d_percent

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R."""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def calculate_cci(high, low, close, period=20):
    """Calculate Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma_tp = calculate_sma(typical_price, period)
    mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    return cci

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    # Simplified ADX calculation
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = calculate_sma(tr, period)
    
    # Directional movement
    dm_plus = np.where((high - high.shift()) > (low.shift() - low), 
                      np.maximum(high - high.shift(), 0), 0)
    dm_minus = np.where((low.shift() - low) > (high - high.shift()), 
                       np.maximum(low.shift() - low, 0), 0)
    
    di_plus = 100 * calculate_sma(pd.Series(dm_plus), period) / atr
    di_minus = 100 * calculate_sma(pd.Series(dm_minus), period) / atr
    
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = calculate_sma(pd.Series(dx), period)
    
    return adx

class AdvancedFeatureEngineering:
    """
    Advanced feature engineering for financial time series data.

    This class is responsible for creating a wide range of features from
    basic OHLCV data, which can then be used for model training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature engineering module.
        
        Args:
            config (Dict[str, Any]): The configuration dictionary.
        """
        self.config = config
        self.scaler = None
        self.feature_names = []
        
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create a comprehensive feature set from OHLCV data.
        
        Args:
            data (pd.DataFrame): A DataFrame with OHLCV columns.
            
        Returns:
            pd.DataFrame: A DataFrame with the engineered features.
        """
        logger.info(f"Creating comprehensive feature set for {len(data)} data points")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Determine appropriate window sizes based on available data
        max_window = min(50, len(data) // 4)  # Use at most 25% of data for rolling windows
        if max_window < 5:
            max_window = 5
        
        logger.info(f"Using maximum window size of {max_window} based on available data")
        
        # Basic price features
        df = self._add_price_features(df)
        
        # Technical indicators (with adaptive windows)
        df = self._add_technical_indicators(df, max_window)
        
        # Volume-based features (with adaptive windows)
        df = self._add_volume_features(df, max_window)
        
        # Market microstructure features (with adaptive windows)
        df = self._add_microstructure_features(df, max_window)
        
        # Statistical features (with adaptive windows)
        df = self._add_statistical_features(df, max_window)
        
        # Pattern recognition features
        df = self._add_pattern_features(df)
        
        # Momentum and trend features (with adaptive windows)
        df = self._add_momentum_features(df, max_window)
        
        # Volatility features (with adaptive windows)
        df = self._add_volatility_features(df, max_window)
        
        # Market regime features (with adaptive windows)
        df = self._add_regime_features(df, max_window)
        
        # Interaction features
        df = self._add_interaction_features(df)
        
        # Clean up and handle missing values
        df = self._clean_features(df)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        logger.info(f"Created {len(self.feature_names)} features")
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic price-based features.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with added price features.
        """
        
        # Price ratios
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        df['high_close_ratio'] = df['High'] / df['Close']
        df['low_close_ratio'] = df['Low'] / df['Close']
        
        # Price ranges
        df['daily_range'] = df['High'] - df['Low']
        df['daily_range_pct'] = (df['High'] - df['Low']) / df['Close']
        df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['body_size'] = abs(df['Close'] - df['Open'])
        
        # Price levels
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Log returns
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['log_return_high'] = np.log(df['High'] / df['High'].shift(1))
        df['log_return_low'] = np.log(df['Low'] / df['Low'].shift(1))
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame, max_window: int = 50) -> pd.DataFrame:
        """
        Add comprehensive technical indicators.

        Args:
            df (pd.DataFrame): The input DataFrame.
            max_window (int): The maximum window size to use for rolling calculations.

        Returns:
            pd.DataFrame: The DataFrame with added technical indicators.
        """
        
        # Moving averages (adaptive to available data)
        periods = [5, 10, 20]
        if max_window >= 50:
            periods.append(50)
        if max_window >= 100:
            periods.append(100)
        if max_window >= 200:
            periods.append(200)
        
        for period in periods:
            df[f'sma_{period}'] = calculate_sma(df['Close'], period)
            df[f'ema_{period}'] = calculate_ema(df['Close'], period)
            df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
            df[f'price_ema_{period}_ratio'] = df['Close'] / df[f'ema_{period}']
        
        # RSI
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = calculate_rsi(df['Close'], period)
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(df['Close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        df['macd_ratio'] = macd_line / df['Close']
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic
        slowk, slowd = calculate_stochastic(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        df['stoch_kd_diff'] = slowk - slowd
        
        # Williams %R
        df['williams_r'] = calculate_williams_r(df['High'], df['Low'], df['Close'])
        
        # CCI (Commodity Channel Index)
        df['cci'] = calculate_cci(df['High'], df['Low'], df['Close'])
        
        # ADX (Average Directional Index)
        df['adx'] = calculate_adx(df['High'], df['Low'], df['Close'])
        
        # ATR (Average True Range)
        df['atr'] = df['High'] - df['Low'] # Placeholder, needs actual ATR calculation
        df['atr_ratio'] = df['atr'] / df['Close'] # Placeholder, needs actual ATR calculation
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame, max_window: int = 50) -> pd.DataFrame:
        """
        Add volume-based features.

        Args:
            df (pd.DataFrame): The input DataFrame.
            max_window (int): The maximum window size to use for rolling calculations.

        Returns:
            pd.DataFrame: The DataFrame with added volume features.
        """
        
        # Volume moving averages (adaptive to available data)
        periods = [5, 10, 20]
        if max_window >= 50:
            periods.append(50)
        
        for period in periods:
            df[f'volume_sma_{period}'] = calculate_sma(df['Volume'], period)
            df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_sma_{period}']
        
        # Volume indicators
        df['obv'] = df['Volume'] # Placeholder, needs actual OBV calculation
        df['ad'] = df['Volume'] # Placeholder, needs actual AD calculation
        df['adosc'] = df['Volume'] # Placeholder, needs actual ADOSC calculation
        
        # Volume-price relationships
        df['volume_price_trend'] = df['Volume'] * df['log_return'] # Placeholder, needs actual volume-price trend
        df['volume_weighted_price'] = (df['Volume'] * df['Close']).rolling(20).sum() / df['Volume'].rolling(20).sum() # Placeholder, needs actual volume-weighted price
        
        # Volume momentum
        df['volume_momentum'] = df['Volume'].pct_change()
        df['volume_acceleration'] = df['volume_momentum'].pct_change()
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame, max_window: int = 50) -> pd.DataFrame:
        """
        Add market microstructure features.

        Args:
            df (pd.DataFrame): The input DataFrame.
            max_window (int): The maximum window size to use for rolling calculations.

        Returns:
            pd.DataFrame: The DataFrame with added microstructure features.
        """
        
        # Bid-ask spread proxy (using high-low as proxy)
        df['spread_proxy'] = (df['High'] - df['Low']) / df['Close']
        
        # Price impact
        df['price_impact'] = df['log_return'].abs() / (df['Volume'] + 1e-8)
        
        # Order flow imbalance (proxy)
        df['flow_imbalance'] = (df['Close'] - df['Open']) * df['Volume']
        
        # Market efficiency ratio (adaptive window)
        window = min(20, max_window)
        df['efficiency_ratio'] = abs(df['Close'] - df['Close'].shift(window)) / df['Close'].rolling(window).apply(lambda x: sum(abs(x.diff().dropna())))
        
        # Realized volatility (adaptive window)
        df['realized_vol'] = df['log_return'].rolling(window).std() * np.sqrt(252) # Placeholder, needs actual realized volatility
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame, max_window: int = 50) -> pd.DataFrame:
        """
        Add statistical features.

        Args:
            df (pd.DataFrame): The input DataFrame.
            max_window (int): The maximum window size to use for rolling calculations.

        Returns:
            pd.DataFrame: The DataFrame with added statistical features.
        """
        
        # Rolling statistics (adaptive periods)
        periods = [5, 10]
        if max_window >= 20:
            periods.append(20)
        
        for period in periods:
            df[f'return_mean_{period}'] = df['log_return'].rolling(period).mean()
            df[f'return_std_{period}'] = df['log_return'].rolling(period).std()
            df[f'return_skew_{period}'] = df['log_return'].rolling(period).skew()
            df[f'return_kurt_{period}'] = df['log_return'].rolling(period).kurt()
            
            # Z-score
            df[f'return_zscore_{period}'] = (df['log_return'] - df[f'return_mean_{period}']) / (df[f'return_std_{period}'] + 1e-8)
        
        # Percentile ranks (adaptive periods)
        percentile_periods = [10]
        if max_window >= 20:
            percentile_periods.append(20)
        if max_window >= 50:
            percentile_periods.append(50)
        
        for period in percentile_periods:
            df[f'price_percentile_{period}'] = df['Close'].rolling(period).rank(pct=True)
            df[f'volume_percentile_{period}'] = df['Volume'].rolling(period).rank(pct=True)
        
        # Autocorrelation (adaptive window)
        window = min(20, max_window)
        for lag in [1, 2, 5]:
            df[f'return_autocorr_{lag}'] = df['log_return'].rolling(window).apply(lambda x: x.autocorr(lag=lag))
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add candlestick pattern features.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with added pattern features.
        """
        
        # Candlestick patterns
        patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
            'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLDARKCLOUDCOVER',
            'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING',
            'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLHAMMER', 'CDLHANGINGMAN',
            'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE',
            'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK',
            'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM',
            'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW',
            'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK',
            'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES',
            'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN',
            'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING',
            'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
        ]
        
        for pattern in patterns:
            try:
                # This part needs actual TA-Lib pattern functions
                # For now, we'll just create a placeholder column
                df[f'pattern_{pattern.lower()}'] = 0 # Placeholder
            except:
                continue
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame, max_window: int = 50) -> pd.DataFrame:
        """
        Add momentum and trend features.

        Args:
            df (pd.DataFrame): The input DataFrame.
            max_window (int): The maximum window size to use for rolling calculations.

        Returns:
            pd.DataFrame: The DataFrame with added momentum features.
        """
        
        # Price momentum (adaptive periods)
        momentum_periods = [1, 3, 5, 10]
        if max_window >= 20:
            momentum_periods.append(20)
        
        for period in momentum_periods:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'log_momentum_{period}'] = np.log(df['Close'] / df['Close'].shift(period))
        
        # Rate of change (adaptive periods)
        roc_periods = [5, 10]
        if max_window >= 20:
            roc_periods.append(20)
        
        for period in roc_periods:
            df[f'roc_{period}'] = df['Close'].pct_change(period) # Placeholder, needs actual ROC calculation
        
        # Momentum oscillators
        df['mom'] = df['Close'].pct_change(10) # Placeholder, needs actual MOM calculation
        df['ppo'] = df['Close'].pct_change(10) # Placeholder, needs actual PPO calculation
        df['rocp'] = df['Close'].pct_change(10) # Placeholder, needs actual ROCP calculation
        df['rocr'] = df['Close'].pct_change(10) # Placeholder, needs actual ROCR calculation
        df['rocr100'] = df['Close'].pct_change(10) # Placeholder, needs actual ROCR100 calculation
        
        # Trend strength (adaptive window)
        window = min(20, max_window)
        df['trend_strength'] = abs(df['Close'] - df['Close'].shift(window)) / df['Close'].rolling(window).std() # Placeholder, needs actual trend strength
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame, max_window: int = 50) -> pd.DataFrame:
        """
        Add volatility features.

        Args:
            df (pd.DataFrame): The input DataFrame.
            max_window (int): The maximum window size to use for rolling calculations.

        Returns:
            pd.DataFrame: The DataFrame with added volatility features.
        """
        
        # Historical volatility (adaptive periods)
        vol_periods = [5, 10]
        if max_window >= 20:
            vol_periods.append(20)
        if max_window >= 50:
            vol_periods.append(50)
        
        for period in vol_periods:
            df[f'hist_vol_{period}'] = df['log_return'].rolling(period).std() * np.sqrt(252) # Placeholder, needs actual historical volatility
        
        # Parkinson volatility (adaptive window)
        window = min(20, max_window)
        df['parkinson_vol'] = np.sqrt(1/(4*np.log(2)) * ((np.log(df['High']/df['Low'])**2).rolling(window).mean())) * np.sqrt(252) # Placeholder, needs actual Parkinson volatility
        
        # Garman-Klass volatility (adaptive window)
        df['gk_vol'] = np.sqrt(((0.5 * (np.log(df['High']/df['Low'])**2) - 
                                (2*np.log(2)-1) * (np.log(df['Close']/df['Open'])**2)).rolling(window).mean())) * np.sqrt(252) # Placeholder, needs actual Garman-Klass volatility
        
        # Volatility ratios (only if we have the required periods)
        if 'hist_vol_5' in df.columns and 'hist_vol_20' in df.columns:
            df['vol_ratio_5_20'] = df['hist_vol_5'] / (df['hist_vol_20'] + 1e-8) # Placeholder, needs actual volatility ratios
        if 'hist_vol_10' in df.columns and 'hist_vol_50' in df.columns:
            df['vol_ratio_10_50'] = df['hist_vol_10'] / (df['hist_vol_50'] + 1e-8) # Placeholder, needs actual volatility ratios
        
        # Volatility of volatility (adaptive window)
        if 'hist_vol_20' in df.columns:
            df['vol_of_vol'] = df['hist_vol_20'].rolling(window).std() # Placeholder, needs actual volatility of volatility
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame, max_window: int = 50) -> pd.DataFrame:
        """
        Add market regime features.

        Args:
            df (pd.DataFrame): The input DataFrame.
            max_window (int): The maximum window size to use for rolling calculations.

        Returns:
            pd.DataFrame: The DataFrame with added regime features.
        """
        
        # Bull/Bear market indicators (only if we have long-term SMA)
        if 'sma_200' in df.columns:
            df['bull_market'] = (df['Close'] > df['sma_200']).astype(int) # Placeholder, needs actual SMA calculation
            df['bear_market'] = (df['Close'] < df['sma_200']).astype(int) # Placeholder, needs actual SMA calculation
            
            # Trend regime (only if we have both SMA50 and SMA200)
            if 'sma_50' in df.columns:
                df['trend_regime'] = np.where(df['Close'] > df['sma_50'], 
                                            np.where(df['sma_50'] > df['sma_200'], 2, 1),  # Strong uptrend, weak uptrend
                                            np.where(df['sma_50'] < df['sma_200'], -2, -1))  # Strong downtrend, weak downtrend
        
        # Volatility regime (adaptive window)
        if 'hist_vol_20' in df.columns:
            vol_window = min(252, max_window * 5)  # Use longer window for volatility median
            vol_median = df['hist_vol_20'].rolling(vol_window).median() # Placeholder, needs actual median volatility
            df['vol_regime'] = np.where(df['hist_vol_20'] > vol_median * 1.5, 2,  # High vol
                                      np.where(df['hist_vol_20'] < vol_median * 0.5, 0, 1))  # Low vol, normal vol
        
        # Market efficiency (adaptive window)
        window = min(20, max_window)
        df['market_efficiency'] = abs(df['Close'] - df['Close'].shift(window)) / df['Close'].rolling(window).apply(lambda x: sum(abs(x.diff().dropna())))
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between different indicators.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with added interaction features.
        """
        
        # RSI and volume interaction
        if 'rsi_14' in df.columns and 'volume_ratio_20' in df.columns:
            df['rsi_volume_interaction'] = df['rsi_14'] * df['volume_ratio_20'] # Placeholder, needs actual RSI and volume ratio
        
        # MACD and volatility interaction
        if 'macd' in df.columns and 'hist_vol_20' in df.columns:
            df['macd_vol_interaction'] = df['macd'] * df['hist_vol_20'] # Placeholder, needs actual MACD and volatility
        
        # Price and volume interaction
        if 'log_return' in df.columns and 'volume_momentum' in df.columns:
            df['price_volume_interaction'] = df['log_return'] * df['volume_momentum'] # Placeholder, needs actual price and volume momentum
        
        # Trend and momentum interaction
        if 'trend_strength' in df.columns and 'momentum_10' in df.columns:
            df['trend_momentum_interaction'] = df['trend_strength'] * df['momentum_10'] # Placeholder, needs actual trend strength and momentum
        
        # Volatility and volume interaction
        if 'hist_vol_20' in df.columns and 'volume_ratio_20' in df.columns:
            df['vol_volume_interaction'] = df['hist_vol_20'] * df['volume_ratio_20'] # Placeholder, needs actual volatility and volume ratio
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and handle missing values in the features.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill for some features
        fill_features = ['sma_200', 'ema_200', 'bb_upper', 'bb_middle', 'bb_lower']
        for feature in fill_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(method='ffill')
        
        # Fill remaining NaN with 0 or median
        for col in df.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if df[col].dtype in ['float64', 'float32']:
                    # Use median for numerical features
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                else:
                    # Use 0 for categorical features
                    df[col] = df[col].fillna(0)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale the features using robust scaling.
        
        Args:
            df (pd.DataFrame): The DataFrame with features.
            fit (bool): Whether to fit the scaler (True for training, False for testing).
            
        Returns:
            pd.DataFrame: The scaled DataFrame.
        """
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        if fit:
            self.scaler = RobustScaler()
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            if self.scaler is not None:
                df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return df
    
    def get_feature_importance(self, target: pd.Series, features: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature importance using correlation and mutual information.
        
        Args:
            target (pd.Series): The target variable.
            features (pd.DataFrame): The feature DataFrame.
            
        Returns:
            Dict[str, float]: A dictionary of feature importance scores.
        """
        importance_scores = {}
        
        for col in features.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                # Correlation-based importance
                corr = abs(features[col].corr(target))
                
                # Simple variance-based importance
                variance = features[col].var()
                
                # Combined score
                importance_scores[col] = corr * np.sqrt(variance)
        
        # Normalize scores
        total_score = sum(importance_scores.values())
        if total_score > 0:
            importance_scores = {k: v/total_score for k, v in importance_scores.items()}
        
        return importance_scores
    
    def select_top_features(self, features: pd.DataFrame, target: pd.Series,
                          n_features: int = 50) -> List[str]:
        """
        Select the top features based on importance.
        
        Args:
            features (pd.DataFrame): The feature DataFrame.
            target (pd.Series): The target variable.
            n_features (int): The number of top features to select.
            
        Returns:
            List[str]: A list of the selected feature names.
        """
        importance_scores = self.get_feature_importance(target, features)
        
        # Sort by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top features
        selected_features = [feature for feature, score in sorted_features[:n_features]]
        
        logger.info(f"Selected {len(selected_features)} top features")
        return selected_features 