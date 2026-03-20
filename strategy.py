#!/usr/bin/env python3
"""
strategy.py - Multi-Timeframe Trend Momentum V3
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #010 success (Sharpe=0.330, Return=+40.7%), improving:
    - Multi-timeframe trend filtering (synthetic 4h trend via long EMAs)
    - Better RSI momentum zone scoring with divergence detection
    - Volume profile confirmation with rolling percentile
    - Adaptive volatility scaling with ATR normalization
    - Signal hysteresis to reduce whipsaws
    
    Key improvements over #010:
    - Higher timeframe trend filter (EMA_200 + EMA_500 for 4h proxy)
    - RSI momentum slope detection (not just absolute value)
    - Volume surge detection (current vs rolling average)
    - Signal hysteresis band to prevent flip-flopping
    - More conservative entry thresholds

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

name = "multi_tf_trend_momentum_v3"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for risk-adjusted returns

# EMA periods for multi-timeframe trend detection
EMA_FAST = 12       # Short-term momentum
EMA_MEDIUM = 26     # Medium-term trend
EMA_SLOW = 50       # Long-term trend
EMA_MAJOR = 200     # 4h proxy trend (200 * 1h = ~8 days)
EMA_SUPER = 500     # Daily proxy trend (500 * 1h = ~21 days)

# RSI configuration with momentum scoring
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_BULL_ZONE = 55   # RSI above this supports longs
RSI_BEAR_ZONE = 45   # RSI below this supports shorts

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_SURGE_THRESHOLD = 1.5  # Volume must be 1.5x average

# Trend scoring weights
WEIGHT_FAST = 0.35
WEIGHT_MEDIUM = 0.35
WEIGHT_SLOW = 0.30

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012  # Target hourly volatility
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.65  # Exponential smoothing factor
HYSTERESIS_BAND = 0.10   # Minimum change to flip signal direction


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


def calculate_rsi_slope(rsi: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Calculate RSI momentum slope using linear regression on recent values.
    Only uses past RSI data (no look-ahead).
    
    Args:
        rsi: Array of RSI values
        lookback: Number of periods for slope calculation
    
    Returns:
        Array of RSI slope values
    """
    n = len(rsi)
    slope = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return slope
    
    for i in range(lookback, n):
        window = rsi[i-lookback:i+1]
        x = np.arange(len(window))
        x_mean = x.mean()
        y_mean = window.mean()
        
        numerator = np.sum((x - x_mean) * (window - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator > 0:
            slope[i] = numerator / denominator
        else:
            slope[i] = 0.0
    
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


def calculate_volume_surge(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume surge ratio (current volume / rolling average).
    Only uses past volume data (no look-ahead).
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for average calculation
    
    Returns:
        Array of volume surge ratios
    """
    n = len(volume)
    volume_surge = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_surge
    
    volume_series = pd.Series(volume)
    
    for i in range(lookback, n):
        window = volume_series.iloc[i-lookback:i]
        avg_volume = window.mean()
        if avg_volume > 0:
            volume_surge[i] = volume[i] / avg_volume
        else:
            volume_surge[i] = 1.0
    
    return volume_surge


def calculate_trend_score(close: float, ema_fast: float, ema_medium: float, 
                          ema_slow: float, ema_major: float, ema_super: float) -> float:
    """
    Calculate multi-timeframe weighted trend score.
    
    Args:
        close: Current close price
        ema_fast: Fast EMA value
        ema_medium: Medium EMA value
        ema_slow: Slow EMA value
        ema_major: Major EMA value (4h proxy)
        ema_super: Super EMA value (daily proxy)
    
    Returns:
        Trend score in range [-1, 1]
    """
    if close <= 0 or ema_super <= 0:
        return 0.0
    
    # Calculate individual trend components (normalized by price)
    fast_score = (ema_fast - ema_medium) / close
    medium_score = (ema_medium - ema_slow) / close
    slow_score = (ema_slow - ema_major) / close
    
    # Weight and combine short/medium/long term trends
    trend_component = (
        WEIGHT_FAST * fast_score +
        WEIGHT_MEDIUM * medium_score +
        WEIGHT_SLOW * slow_score
    )
    
    # Major trend alignment bonus (4h proxy)
    major_alignment = np.sign(ema_major - ema_super)
    major_bonus = 0.3 if major_alignment > 0 else -0.3
    
    # Super trend filter (daily proxy) - only trade in direction of major trend
    super_direction = np.sign(close - ema_super)
    
    # Combine all components
    trend_score = trend_component * 100 + major_bonus
    trend_score = trend_score * (1.0 + 0.5 * super_direction)
    
    # Normalize to [-1, 1] range
    trend_score = np.clip(trend_score, -1.0, 1.0)
    
    return trend_score


def calculate_rsi_momentum_score(rsi: float, rsi_slope: float) -> float:
    """
    Calculate RSI momentum score combining absolute value and slope.
    
    Args:
        rsi: Current RSI value
        rsi_slope: RSI slope (change per bar)
    
    Returns:
        RSI momentum score in range [0, 1]
    """
    # Base score from RSI zone
    if rsi < RSI_OVERSOLD:
        base_score = 0.2  # Oversold but risky
    elif rsi < RSI_BEAR_ZONE:
        base_score = 0.4  # Bearish zone
    elif rsi < RSI_BULL_ZONE:
        base_score = 0.5  # Neutral
    elif rsi < RSI_OVERBOUGHT:
        base_score = 0.7  # Bullish zone
    else:
        base_score = 0.5  # Overbought, reduce confidence
    
    # Slope adjustment (momentum confirmation)
    slope_factor = np.clip(rsi_slope * 5, -0.3, 0.3)  # Normalize slope impact
    
    # Combine base score with momentum
    momentum_score = base_score + slope_factor
    momentum_score = np.clip(momentum_score, 0.0, 1.0)
    
    return momentum_score


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Multi-Timeframe Trend Momentum V3 Strategy.
    
    Signal Logic:
    1. Multi-timeframe trend score (1h/4h/daily proxy via EMAs)
    2. RSI momentum with slope detection
    3. Volume surge confirmation
    4. Volatility-based position sizing
    5. Signal smoothing with hysteresis
    
    Entry Conditions:
    - LONG: Positive trend + RSI > 45 + volume surge + aligned higher TF
    - SHORT: Negative trend + RSI < 55 + volume surge + aligned higher TF
    
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
    ema_super = calculate_ema(close, EMA_SUPER)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    rsi_slope = calculate_rsi_slope(rsi, lookback=5)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_surge = calculate_volume_surge(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index (need enough data for all indicators)
    min_valid_index = max(
        EMA_SUPER,
        RSI_PERIOD + 6,  # RSI + slope lookback
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK
    )
    
    # Track previous signal for smoothing and hysteresis
    prev_signal = 0.0
    prev_direction = 0  # 0=neutral, 1=long, -1=short
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Check volatility regime (avoid extreme volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate multi-timeframe trend score
        trend_score = calculate_trend_score(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i], ema_super[i]
        )
        
        # Skip weak trends
        if abs(trend_score) < 0.20:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate RSI momentum score
        rsi_momentum = calculate_rsi_momentum_score(rsi[i], rsi_slope[i])
        
        # Volume surge confirmation
        volume_confirmed = volume_surge[i] >= VOLUME_SURGE_THRESHOLD
        
        # Higher timeframe alignment check
        ht_aligned = np.sign(ema_major[i] - ema_super[i]) == np.sign(trend_score)
        
        # Determine signal direction and base magnitude
        if trend_score > 0:
            # LONG bias
            if rsi[i] < RSI_BEAR_ZONE:
                base_signal = 0.0  # RSI too weak for long
            elif not ht_aligned:
                base_signal = trend_score * 0.5  # Reduce size if HT not aligned
            else:
                base_signal = trend_score * rsi_momentum
        else:
            # SHORT bias
            if rsi[i] > RSI_BULL_ZONE:
                base_signal = 0.0  # RSI too strong for short
            elif not ht_aligned:
                base_signal = trend_score * 0.5  # Reduce size if HT not aligned
            else:
                base_signal = trend_score * (1.0 - rsi_momentum * 0.5)
        
        # Apply volume confirmation boost
        if volume_confirmed:
            base_signal *= 1.20
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.5)
        
        raw_signal = base_signal * vol_factor
        
        # Apply hysteresis (prevent flip-flopping)
        current_direction = np.sign(raw_signal)
        if current_direction != 0 and current_direction != prev_direction:
            # Only flip if signal change exceeds hysteresis band
            if abs(raw_signal - prev_signal) < HYSTERESIS_BAND:
                raw_signal = prev_signal * 0.5  # Reduce instead of flip
        
        # Apply exponential smoothing to reduce whipsaws
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        prev_signal = smoothed_signal
        prev_direction = np.sign(smoothed_signal)
        
        # Apply thresholds
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals