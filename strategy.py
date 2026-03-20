#!/usr/bin/env python3
"""
strategy.py - Trend Momentum V3 with Volatility Regime Filter
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on trend_momentum_v2 (Sharpe=0.330), improving:
    - Bollinger Band squeeze detection to avoid choppy markets
    - Better trend strength filtering (only trade strong trends)
    - RSI momentum with zone-based scoring
    - More conservative signal thresholds to reduce whipsaws
    - Improved volatility-based position sizing

Key improvements over v2:
    - BB squeeze filter (avoid low volatility periods)
    - Trend strength threshold (minimum trend score required)
    - RSI momentum confirmation with better zone handling
    - Signal decay mechanism to prevent overtrading
    - Adaptive position sizing based on volatility regime

Look-Ahead Safety:
    - All rolling calculations use only past data (min_periods respected)
    - No .shift(-n) or future index access
    - Signal at bar t uses only prices.iloc[:t+1]
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "trend_momentum_v3"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for better risk-adjusted returns

# EMA periods for trend detection
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_BULL_ZONE = 50  # Above this = bullish momentum
RSI_BEAR_ZONE = 50  # Below this = bearish momentum

# Bollinger Band configuration
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 0.005  # Bandwidth below this = squeeze

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_PERCENTILE_THRESHOLD = 0.5  # Volume must be above median

# Trend scoring
WEIGHT_FAST = 0.35
WEIGHT_MEDIUM = 0.35
WEIGHT_SLOW = 0.30
MIN_TREND_STRENGTH = 0.20  # Minimum trend score to trade

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012  # Target hourly volatility
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.6  # Exponential smoothing factor (0-1)


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    
    Args:
        close: Array of close prices
        period: EMA period
    
    Returns:
        Array of EMA values
    """
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return ema
    
    close_series = pd.Series(close)
    ema_values = close_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    ema = np.nan_to_num(ema_values, nan=0.0)
    
    return ema


def calculate_sma(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using only past data.
    
    Args:
        close: Array of close prices
        period: SMA period
    
    Returns:
        Array of SMA values
    """
    n = len(close)
    sma = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return sma
    
    close_series = pd.Series(close)
    sma_values = close_series.rolling(window=period, min_periods=period).mean().values
    sma = np.nan_to_num(sma_values, nan=0.0)
    
    return sma


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using only past data.
    
    Args:
        close: Array of close prices
        period: RSI period
    
    Returns:
        Array of RSI values (0-100)
    """
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    close_series = pd.Series(close)
    delta = close_series.diff()
    
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.nan_to_num(rsi_series.values, nan=50.0)
    
    return rsi


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using only past data.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ATR period
    
    Returns:
        Array of ATR values
    """
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    tr_series = pd.Series(tr)
    atr_series = tr_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    atr = np.nan_to_num(atr_series.values, nan=0.0)
    
    return atr


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands using only past data.
    
    Args:
        close: Array of close prices
        period: BB period
        std_dev: Number of standard deviations
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band, bandwidth)
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    bandwidth = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower, bandwidth
    
    close_series = pd.Series(close)
    middle_series = close_series.rolling(window=period, min_periods=period).mean()
    std_series = close_series.rolling(window=period, min_periods=period).std()
    
    upper_series = middle_series + (std_dev * std_series)
    lower_series = middle_series - (std_dev * std_series)
    bandwidth_series = (upper_series - lower_series) / middle_series
    
    upper = np.nan_to_num(upper_series.values, nan=0.0)
    middle = np.nan_to_num(middle_series.values, nan=0.0)
    lower = np.nan_to_num(lower_series.values, nan=0.0)
    bandwidth = np.nan_to_num(bandwidth_series.values, nan=0.0)
    
    return upper, middle, lower, bandwidth


def calculate_volume_percentile(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume percentile rank using rolling window.
    Only uses past volume data (no look-ahead).
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for percentile calculation
    
    Returns:
        Array of volume percentile ranks (0-1)
    """
    n = len(volume)
    volume_pct = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return volume_pct
    
    volume_series = pd.Series(volume)
    
    for i in range(lookback, n):
        window = volume_series.iloc[i-lookback:i]
        rank = (window < volume[i]).sum() / lookback
        volume_pct[i] = rank
    
    return volume_pct


def calculate_trend_score(close: float, ema_fast: float, ema_medium: float, 
                          ema_slow: float, ema_major: float) -> float:
    """
    Calculate weighted trend score based on EMA stack alignment.
    
    Args:
        close: Current close price
        ema_fast: Fast EMA value
        ema_medium: Medium EMA value
        ema_slow: Slow EMA value
        ema_major: Major EMA value
    
    Returns:
        Trend score in range [-1, 1]
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Calculate individual trend components (normalized by price)
    fast_score = (ema_fast - ema_medium) / close
    medium_score = (ema_medium - ema_slow) / close
    slow_score = (ema_slow - ema_major) / close
    
    # Weight and combine
    trend_component = (
        WEIGHT_FAST * fast_score +
        WEIGHT_MEDIUM * medium_score +
        WEIGHT_SLOW * slow_score
    )
    
    # Major trend filter
    major_direction = np.sign(close - ema_major)
    trend_score = trend_component * (1.0 + 0.3 * major_direction)
    
    # Normalize to [-1, 1] range
    trend_score = np.clip(trend_score / 0.015, -1.0, 1.0)
    
    return trend_score


def calculate_rsi_momentum_score(rsi: float) -> float:
    """
    Calculate RSI momentum score (-1 to 1 scale).
    Positive = bullish momentum, Negative = bearish momentum
    
    Args:
        rsi: Current RSI value
    
    Returns:
        RSI momentum score in range [-1, 1]
    """
    if rsi < RSI_OVERSOLD:
        return -0.3  # Oversold, potential reversal up
    elif rsi < RSI_BEAR_ZONE:
        return -0.5  # Bearish momentum
    elif rsi < RSI_BULL_ZONE:
        return 0.0  # Neutral
    elif rsi < RSI_OVERBOUGHT:
        return 0.5  # Bullish momentum
    else:
        return 0.3  # Overbought, potential reversal down


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum V3 Strategy with Volatility Regime Filter.
    
    Signal Logic:
    1. Weighted trend score from EMA stack alignment
    2. Bollinger Band squeeze filter (avoid choppy markets)
    3. RSI momentum scoring
    4. Volume percentile ranking for confirmation
    5. Volatility-based position sizing
    6. Signal smoothing to reduce whipsaws
    
    Entry Conditions:
    - LONG: Positive trend score + BB not squeezing + RSI momentum positive
    - SHORT: Negative trend score + BB not squeezing + RSI momentum negative
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract required columns with safety checks
    try:
        close = prices["close"].values.astype(np.float64)
        high = prices["high"].values.astype(np.float64)
        low = prices["low"].values.astype(np.float64)
        volume = prices["volume"].values.astype(np.float64)
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Handle NaN values
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Ensure valid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_medium = calculate_ema(close, EMA_MEDIUM)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    bb_upper, bb_middle, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    volume_pct = calculate_volume_percentile(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        BB_PERIOD,
        VOLUME_LOOKBACK
    )
    
    # Track previous signal for smoothing
    prev_signal = 0.0
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0 or bb_bandwidth[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check volatility regime (avoid extreme volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check Bollinger Band squeeze (avoid choppy markets)
        if bb_bandwidth[i] < BB_SQUEEZE_THRESHOLD:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate weighted trend score
        trend_score = calculate_trend_score(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i]
        )
        
        # Skip weak trends (require minimum trend strength)
        if abs(trend_score) < MIN_TREND_STRENGTH:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate RSI momentum score
        rsi_momentum = calculate_rsi_momentum_score(rsi[i])
        
        # Volume confirmation
        volume_confirmed = volume_pct[i] >= VOLUME_PERCENTILE_THRESHOLD
        
        # Determine signal direction and base magnitude
        if trend_score > 0:
            # LONG bias - require RSI momentum support
            if rsi_momentum < 0.0:
                base_signal = 0.0  # RSI momentum not supporting long
            else:
                base_signal = trend_score * (0.5 + 0.5 * rsi_momentum)
        else:
            # SHORT bias - require RSI momentum support
            if rsi_momentum > 0.0:
                base_signal = 0.0  # RSI momentum not supporting short
            else:
                base_signal = trend_score * (0.5 - 0.5 * rsi_momentum)
        
        # Apply volume confirmation boost
        if volume_confirmed:
            base_signal *= 1.1
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.8)
        
        raw_signal = base_signal * vol_factor
        
        # Apply exponential smoothing to reduce whipsaws
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        prev_signal = smoothed_signal
        
        # Apply thresholds
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals