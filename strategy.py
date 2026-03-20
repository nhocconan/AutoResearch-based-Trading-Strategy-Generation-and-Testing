#!/usr/bin/env python3
"""
strategy.py - Trend Momentum V3 with Bollinger Volatility Filter
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #010 success (Sharpe=0.330, Return=+40.7%), improving:
    - Bollinger Band width for volatility regime detection
    - Better RSI zone handling (avoid overbought/oversold extremes)
    - Improved volume momentum confirmation
    - Cleaner signal smoothing with adaptive decay
    - More robust edge case handling
    
    Key improvements over #010:
    - BB width filter to avoid trading during extreme volatility
    - RSI momentum slope instead of static zones
    - Volume momentum (change) instead of percentile
    - Adaptive smoothing based on signal confidence
    - Better normalization of trend scores

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
EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration with momentum scoring
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_NEUTRAL_LOW = 45
RSI_NEUTRAL_HIGH = 55

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_MOMENTUM_THRESHOLD = 1.2  # Volume must be 20% above average

# Bollinger Band configuration
BB_PERIOD = 20
BB_STD = 2.0
BB_WIDTH_MIN = 0.02  # Minimum BB width (% of price)
BB_WIDTH_MAX = 0.15  # Maximum BB width (% of price)

# Trend scoring weights
WEIGHT_FAST = 0.4
WEIGHT_MEDIUM = 0.35
WEIGHT_SLOW = 0.25

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012  # Target hourly volatility
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL = 0.12
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.65  # Exponential smoothing factor (0-1)


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


def calculate_rsi_slope(rsi: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Calculate RSI momentum slope using linear regression.
    Only uses past RSI values (no look-ahead).
    
    Args:
        rsi: Array of RSI values
        lookback: Lookback period for slope calculation
    
    Returns:
        Array of RSI slope values
    """
    n = len(rsi)
    rsi_slope = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return rsi_slope
    
    for i in range(lookback, n):
        window = rsi[i-lookback:i]
        x = np.arange(lookback)
        if np.std(window) > 0:
            slope = np.polyfit(x, window, 1)[0]
            rsi_slope[i] = slope
        else:
            rsi_slope[i] = 0.0
    
    return rsi_slope


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
        Tuple of (upper_band, middle_band, lower_band)
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower
    
    close_series = pd.Series(close)
    middle_series = close_series.rolling(window=period, min_periods=period).mean()
    std_series = close_series.rolling(window=period, min_periods=period).std()
    
    middle = np.nan_to_num(middle_series.values, nan=0.0)
    std_vals = np.nan_to_num(std_series.values, nan=0.0)
    
    upper = middle + (std_dev * std_vals)
    lower = middle - (std_dev * std_vals)
    
    return upper, middle, lower


def calculate_bb_width(upper: np.ndarray, lower: np.ndarray, middle: np.ndarray) -> np.ndarray:
    """
    Calculate Bollinger Band width as percentage of middle band.
    
    Args:
        upper: Upper band values
        lower: Lower band values
        middle: Middle band values
    
    Returns:
        Array of BB width values (as % of middle)
    """
    n = len(middle)
    bb_width = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if middle[i] > 0:
            bb_width[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bb_width[i] = 0.0
    
    return bb_width


def calculate_volume_momentum(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume momentum (current vs average).
    Only uses past volume data (no look-ahead).
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for average calculation
    
    Returns:
        Array of volume momentum ratios
    """
    n = len(volume)
    vol_momentum = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return vol_momentum
    
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    
    vol_momentum = np.nan_to_num(volume / avg_volume.values, nan=1.0)
    vol_momentum = np.where(vol_momentum <= 0, 1.0, vol_momentum)
    
    return vol_momentum


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
    major_score = (close - ema_major) / close
    
    # Weight and combine
    trend_component = (
        WEIGHT_FAST * fast_score +
        WEIGHT_MEDIUM * medium_score +
        WEIGHT_SLOW * slow_score
    )
    
    # Major trend filter (amplifies signal in direction of major trend)
    major_direction = np.sign(major_score)
    trend_score = trend_component * (1.0 + 0.5 * major_direction)
    
    # Normalize to [-1, 1] range
    trend_score = np.clip(trend_score / 0.015, -1.0, 1.0)
    
    return trend_score


def calculate_rsi_momentum_score(rsi: float, rsi_slope: float) -> float:
    """
    Calculate RSI momentum score based on RSI value and slope.
    
    Args:
        rsi: Current RSI value
        rsi_slope: RSI slope (change per bar)
    
    Returns:
        RSI momentum score in range [0, 1]
    """
    # Base score from RSI value
    if rsi <= RSI_OVERSOLD:
        base_score = 0.2  # Oversold but potential reversal risk
    elif rsi < RSI_NEUTRAL_LOW:
        base_score = 0.5  # Recovering zone
    elif rsi < RSI_NEUTRAL_HIGH:
        base_score = 0.7  # Neutral zone
    elif rsi < RSI_OVERBOUGHT:
        base_score = 0.9  # Strong momentum
    else:
        base_score = 0.6  # Overbought, reduce confidence
    
    # Adjust for RSI slope (momentum direction)
    slope_adjustment = np.clip(rsi_slope / 5.0, -0.2, 0.2)
    
    final_score = base_score + slope_adjustment
    final_score = np.clip(final_score, 0.0, 1.0)
    
    return final_score


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum V3 Strategy with Bollinger Volatility Filter.
    
    Signal Logic:
    1. Weighted trend score from EMA stack alignment
    2. RSI momentum scoring with slope detection
    3. Volume momentum confirmation
    4. Bollinger Band width filter for volatility regime
    5. ATR-based position sizing
    6. Adaptive signal smoothing
    
    Entry Conditions:
    - LONG: Positive trend score + RSI momentum > 0.5 + volume confirmation + BB width OK
    - SHORT: Negative trend score + RSI momentum < 0.7 + volume confirmation + BB width OK
    
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
    rsi_slope = calculate_rsi_slope(rsi, lookback=5)
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_middle)
    
    vol_momentum = calculate_volume_momentum(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        BB_PERIOD,
        VOLUME_LOOKBACK + 5
    )
    
    # Track previous signal for smoothing
    prev_signal = 0.0
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0 or bb_middle[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check Bollinger Band width (volatility regime filter)
        bb_width_pct = bb_width[i]
        if bb_width_pct < BB_WIDTH_MIN or bb_width_pct > BB_WIDTH_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check ATR volatility regime
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate weighted trend score
        trend_score = calculate_trend_score(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i]
        )
        
        # Skip weak trends
        if abs(trend_score) < 0.12:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate RSI momentum score
        rsi_momentum_score = calculate_rsi_momentum_score(rsi[i], rsi_slope[i])
        
        # Volume confirmation
        volume_confirmed = vol_momentum[i] >= VOLUME_MOMENTUM_THRESHOLD
        
        # Determine signal direction and base magnitude
        if trend_score > 0:
            # LONG bias
            if rsi_momentum_score < 0.45:
                base_signal = 0.0  # RSI momentum too weak for long
            else:
                base_signal = trend_score * rsi_momentum_score
        else:
            # SHORT bias
            if rsi_momentum_score > 0.75:
                base_signal = 0.0  # RSI momentum too strong for short
            else:
                base_signal = trend_score * (1.0 - rsi_momentum_score * 0.4)
        
        # Skip if base signal is zero
        if base_signal == 0.0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Apply volume confirmation boost
        if volume_confirmed:
            base_signal *= 1.20
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.6, 1.8)
        
        raw_signal = base_signal * vol_factor
        
        # Adaptive smoothing based on signal confidence
        confidence = abs(raw_signal)
        adaptive_smooth = SMOOTHING_FACTOR + (0.15 * confidence)
        adaptive_smooth = np.clip(adaptive_smooth, 0.5, 0.85)
        
        smoothed_signal = adaptive_smooth * prev_signal + (1.0 - adaptive_smooth) * raw_signal
        prev_signal = smoothed_signal
        
        # Apply thresholds
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals