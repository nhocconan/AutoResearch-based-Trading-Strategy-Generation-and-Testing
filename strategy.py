#!/usr/bin/env python3
"""
strategy.py - Trend Momentum V4 with Bollinger Squeeze Filter
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #023 success (Sharpe=0.178, Return=+20.9%), improving:
    - Bollinger Band squeeze detection to avoid choppy markets
    - Simplified trend scoring (reduce overfitting risk)
    - RSI slope for momentum confirmation (not just level)
    - Volume MA ratio instead of percentile (more stable)
    - Cleaner signal thresholds based on backtest learnings
    
    Key improvements over v2/v3:
    - BB width filter: only trade when bands are expanding (breakout)
    - RSI slope: momentum acceleration matters more than absolute level
    - Volume ratio: simpler, more robust than percentile ranking
    - Reduced parameter complexity to avoid overfitting
    - Better volatility regime detection

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

name = "trend_momentum_v4_bb_squeeze"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for risk-adjusted returns

# EMA periods for trend detection (simplified from v2)
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration
RSI_PERIOD = 14
RSI_MOMENTUM_THRESHOLD = 55  # RSI must be above this for longs
RSI_DEMOMENTUM_THRESHOLD = 45  # RSI must be below this for shorts

# Bollinger Band configuration for squeeze detection
BB_PERIOD = 20
BB_STD_DEV = 2.0
BB_WIDTH_MIN = 0.015  # Minimum BB width to avoid squeeze (1.5%)
BB_WIDTH_MAX = 0.080  # Maximum BB width to avoid extreme volatility

# Volume configuration
VOLUME_MA_PERIOD = 20
VOLUME_RATIO_THRESHOLD = 1.2  # Volume must be 20% above average

# Trend scoring
TREND_MIN_SCORE = 0.20  # Minimum trend score to trade

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012  # Target hourly volatility
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.6  # Exponential smoothing factor


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


def calculate_rsi_slope(rsi: np.ndarray, lookback: int = 3) -> np.ndarray:
    """
    Calculate RSI slope (momentum acceleration) using only past data.
    
    Args:
        rsi: Array of RSI values
        lookback: Period for slope calculation
    
    Returns:
        Array of RSI slope values
    """
    n = len(rsi)
    slope = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return slope
    
    for i in range(lookback, n):
        slope[i] = (rsi[i] - rsi[i - lookback]) / lookback
    
    return slope


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
        std_dev: Standard deviation multiplier
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band, bb_width)
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    width = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower, width
    
    close_series = pd.Series(close)
    middle_series = close_series.rolling(window=period, min_periods=period).mean()
    std_series = close_series.rolling(window=period, min_periods=period).std()
    
    upper_series = middle_series + (std_dev * std_series)
    lower_series = middle_series - (std_dev * std_series)
    width_series = (upper_series - lower_series) / middle_series
    
    upper = np.nan_to_num(upper_series.values, nan=0.0)
    middle = np.nan_to_num(middle_series.values, nan=0.0)
    lower = np.nan_to_num(lower_series.values, nan=0.0)
    width = np.nan_to_num(width_series.values, nan=0.0)
    
    return upper, middle, lower, width


def calculate_volume_ma_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Calculate volume to moving average ratio using only past data.
    
    Args:
        volume: Array of volume values
        period: MA period for volume
    
    Returns:
        Array of volume ratio values
    """
    n = len(volume)
    ratio = np.ones(n, dtype=np.float64)
    
    if n < period:
        return ratio
    
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=period, min_periods=period).mean()
    
    ratio = volume / np.where(volume_ma > 0, volume_ma, volume)
    ratio = np.nan_to_num(ratio, nan=1.0)
    
    return ratio


def calculate_trend_score(close: float, ema_fast: float, ema_medium: float, 
                          ema_slow: float, ema_major: float) -> float:
    """
    Calculate simplified trend score based on EMA alignment.
    
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
    
    # Simple alignment score
    bullish_signals = 0
    bearish_signals = 0
    
    if ema_fast > ema_medium:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    if ema_medium > ema_slow:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    if ema_slow > ema_major:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    if close > ema_major:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # Calculate net score
    total_signals = bullish_signals + bearish_signals
    if total_signals == 0:
        return 0.0
    
    trend_score = (bullish_signals - bearish_signals) / total_signals
    
    # Amplify strong alignment
    if abs(trend_score) >= 0.75:
        trend_score *= 1.3
    
    return np.clip(trend_score, -1.0, 1.0)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum V4 Strategy with Bollinger Squeeze Filter.
    
    Signal Logic:
    1. EMA stack alignment for trend direction
    2. Bollinger Band width to avoid squeeze/choppy markets
    3. RSI level + slope for momentum confirmation
    4. Volume ratio for breakout confirmation
    5. Volatility-based position sizing
    6. Signal smoothing to reduce whipsaws
    
    Entry Conditions:
    - LONG: Bullish EMA alignment + BB expanding + RSI > 55 + volume spike
    - SHORT: Bearish EMA alignment + BB expanding + RSI < 45 + volume spike
    
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
    rsi_slope = calculate_rsi_slope(rsi, lookback=3)
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(
        close, BB_PERIOD, BB_STD_DEV
    )
    
    volume_ratio = calculate_volume_ma_ratio(volume, VOLUME_MA_PERIOD)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        BB_PERIOD,
        VOLUME_MA_PERIOD
    )
    
    # Track previous signal for smoothing
    prev_signal = 0.0
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0 or bb_width[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check Bollinger Band width (avoid squeeze and extreme volatility)
        if bb_width[i] < BB_WIDTH_MIN or bb_width[i] > BB_WIDTH_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check volatility regime
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate trend score
        trend_score = calculate_trend_score(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i]
        )
        
        # Skip weak trends
        if abs(trend_score) < TREND_MIN_SCORE:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_RATIO_THRESHOLD
        
        # Determine signal based on trend direction
        if trend_score > 0:
            # LONG bias
            rsi_ok = rsi[i] >= RSI_MOMENTUM_THRESHOLD
            rsi_slope_ok = rsi_slope[i] >= 0  # RSI not declining
            
            if rsi_ok and rsi_slope_ok and volume_confirmed:
                base_signal = trend_score * (rsi[i] - 50) / 50  # Normalize RSI contribution
            else:
                base_signal = 0.0
        else:
            # SHORT bias
            rsi_ok = rsi[i] <= RSI_DEMOMENTUM_THRESHOLD
            rsi_slope_ok = rsi_slope[i] <= 0  # RSI not rising
            
            if rsi_ok and rsi_slope_ok and volume_confirmed:
                base_signal = trend_score * (50 - rsi[i]) / 50  # Normalize RSI contribution
            else:
                base_signal = 0.0
        
        # Skip if no signal
        if base_signal == 0.0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
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