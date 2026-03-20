#!/usr/bin/env python3
"""
strategy.py - Volatility Trend Breakout V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #010 success (Sharpe=0.330, Return=+40.7%), improving:
    - Bollinger Band squeeze detection for volatility regimes
    - Breakout confirmation with volume spike
    - Multi-EMA trend alignment filter
    - RSI momentum quality scoring
    - Volatility-based position sizing
    
    Key improvements over #010:
    - Explicit volatility regime detection (squeeze vs expansion)
    - Breakout-only entries during expansion phase
    - Stronger trend filter (4-EMA stack)
    - Volume spike confirmation for breakouts
    - More conservative during squeeze periods

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

name = "volatility_trend_breakout_v1"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for better risk-adjusted returns

# EMA periods for trend detection (multi-timeframe alignment)
EMA_FAST = 8
EMA_MEDIUM = 21
EMA_SLOW = 55
EMA_MAJOR = 200

# Bollinger Band configuration for volatility regime
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 0.015  # Bandwidth below this = squeeze
BB_EXPANSION_THRESHOLD = 0.025  # Bandwidth above this = expansion

# RSI configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_BULLISH_ZONE = 50  # Above this = bullish momentum
RSI_BEARISH_ZONE = 50  # Below this = bearish momentum

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_THRESHOLD = 1.5  # Volume must be 1.5x average for breakout confirmation

# Trend scoring weights
WEIGHT_FAST = 0.35
WEIGHT_MEDIUM = 0.30
WEIGHT_SLOW = 0.25
WEIGHT_MAJOR = 0.10

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012  # Target hourly volatility
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.6  # Exponential smoothing factor (0-1)
BREAKOUT_CONFIRMATION_BARS = 2  # Require N bars of confirmation


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


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands using only past data.
    
    Args:
        close: Array of close prices
        period: BB period
        std_dev: Standard deviation multiplier
    
    Returns:
        Tuple of (upper, middle, lower) band arrays
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
    
    upper_series = middle_series + (std_dev * std_series)
    lower_series = middle_series - (std_dev * std_series)
    
    upper = np.nan_to_num(upper_series.values, nan=0.0)
    middle = np.nan_to_num(middle_series.values, nan=0.0)
    lower = np.nan_to_num(lower_series.values, nan=0.0)
    
    return upper, middle, lower


def calculate_bandwidth(upper: np.ndarray, lower: np.ndarray, middle: np.ndarray) -> np.ndarray:
    """
    Calculate Bollinger Band bandwidth (volatility measure).
    
    Args:
        upper: Upper band values
        lower: Lower band values
        middle: Middle band values
    
    Returns:
        Array of bandwidth values
    """
    bandwidth = np.zeros(len(upper), dtype=np.float64)
    
    # Avoid division by zero
    mask = middle > 0
    bandwidth[mask] = (upper[mask] - lower[mask]) / middle[mask]
    
    return bandwidth


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


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio (current volume / average volume) using rolling window.
    Only uses past volume data (no look-ahead).
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for average calculation
    
    Returns:
        Array of volume ratios
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    
    volume_ratio = volume / avg_volume
    volume_ratio = np.nan_to_num(volume_ratio, nan=1.0)
    
    return volume_ratio


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
        WEIGHT_SLOW * slow_score +
        WEIGHT_MAJOR * major_score
    )
    
    # Normalize to [-1, 1] range (typical values are small)
    trend_score = np.clip(trend_component / 0.008, -1.0, 1.0)
    
    return trend_score


def calculate_rsi_momentum_score(rsi: float, trend_direction: int) -> float:
    """
    Calculate RSI momentum score based on trend direction.
    
    Args:
        rsi: Current RSI value
        trend_direction: 1 for long, -1 for short
    
    Returns:
        RSI momentum score in range [0, 1]
    """
    if trend_direction > 0:
        # Long bias: want RSI above 50 but not overbought
        if rsi < RSI_BEARISH_ZONE:
            return 0.2  # Weak momentum
        elif rsi < RSI_OVERBOUGHT:
            return 0.8  # Good momentum
        else:
            return 0.4  # Overbought, reduce confidence
    else:
        # Short bias: want RSI below 50 but not oversold
        if rsi > RSI_BULLISH_ZONE:
            return 0.2  # Weak momentum
        elif rsi > RSI_OVERSOLD:
            return 0.8  # Good momentum
        else:
            return 0.4  # Oversold, reduce confidence


def detect_volatility_regime(bandwidth: float, squeeze_thresh: float, expansion_thresh: float) -> str:
    """
    Detect volatility regime based on Bollinger Band bandwidth.
    
    Args:
        bandwidth: Current bandwidth value
        squeeze_thresh: Threshold for squeeze regime
        expansion_thresh: Threshold for expansion regime
    
    Returns:
        Regime string: "squeeze", "expansion", or "neutral"
    """
    if bandwidth < squeeze_thresh:
        return "squeeze"
    elif bandwidth > expansion_thresh:
        return "expansion"
    else:
        return "neutral"


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Volatility Trend Breakout V1 Strategy.
    
    Signal Logic:
    1. Detect volatility regime using Bollinger Band bandwidth
    2. Wait for breakout during expansion phase
    3. Confirm with volume spike
    4. Filter by multi-EMA trend alignment
    5. Use RSI for momentum quality
    6. Apply volatility-based position sizing
    
    Entry Conditions:
    - LONG: Expansion regime + price > upper BB + volume spike + bullish trend + RSI > 50
    - SHORT: Expansion regime + price < lower BB + volume spike + bearish trend + RSI < 50
    
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
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    bandwidth = calculate_bandwidth(bb_upper, bb_lower, bb_middle)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        BB_PERIOD,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK
    )
    
    # Track previous signal for smoothing
    prev_signal = 0.0
    prev_regime = "neutral"
    breakout_count = 0
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0 or bb_middle[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_regime = "neutral"
            breakout_count = 0
            continue
        
        # Check volatility regime
        current_regime = detect_volatility_regime(
            bandwidth[i], BB_SQUEEZE_THRESHOLD, BB_EXPANSION_THRESHOLD
        )
        
        # Check volatility limits
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_regime = current_regime
            breakout_count = 0
            continue
        
        # Calculate weighted trend score
        trend_score = calculate_trend_score(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i]
        )
        
        # Determine trend direction
        trend_direction = np.sign(trend_score)
        
        # Skip weak trends
        if abs(trend_score) < 0.20:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_regime = current_regime
            breakout_count = 0
            continue
        
        # Calculate RSI momentum score
        rsi_momentum = calculate_rsi_momentum_score(rsi[i], trend_direction)
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_SPIKE_THRESHOLD
        
        # Detect breakout
        breakout_long = close[i] > bb_upper[i]
        breakout_short = close[i] < bb_lower[i]
        
        # Track breakout confirmation
        if current_regime == "expansion":
            if breakout_long and trend_direction > 0:
                breakout_count += 1
            elif breakout_short and trend_direction < 0:
                breakout_count += 1
            else:
                breakout_count = max(0, breakout_count - 1)
        else:
            breakout_count = 0
        
        # Require breakout confirmation
        if breakout_count < BREAKOUT_CONFIRMATION_BARS:
            signals[i] = 0.0
            prev_signal = prev_signal * 0.9  # Decay previous signal
            prev_regime = current_regime
            continue
        
        # Determine base signal magnitude
        if trend_direction > 0 and breakout_long:
            # LONG signal
            base_signal = abs(trend_score) * rsi_momentum
            if volume_confirmed:
                base_signal *= 1.2
        elif trend_direction < 0 and breakout_short:
            # SHORT signal
            base_signal = -abs(trend_score) * rsi_momentum
            if volume_confirmed:
                base_signal *= 1.2
        else:
            base_signal = 0.0
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        
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
        prev_regime = current_regime
    
    return signals