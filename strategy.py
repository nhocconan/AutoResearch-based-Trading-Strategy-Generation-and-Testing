#!/usr/bin/env python3
"""
strategy.py - Trend Momentum V3 with ADX Filter and Pullback Entries
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #010 success (Sharpe=0.330, Return=+40.7%), improving:
    - ADX filter to distinguish trending vs ranging markets
    - Pullback entry mechanism for better risk/reward
    - Improved RSI directional scoring (bullish vs bearish momentum)
    - Dynamic volatility thresholds based on recent ATR distribution
    - Reduced signal whipsaws through stricter trend confirmation
    
    Key improvements over trend_momentum_v2:
    - ADX threshold (>=25) to filter low-trend environments
    - Pullback detection using EMA distance for entry timing
    - Asymmetric RSI scoring based on trend direction
    - Adaptive volatility percentiles instead of fixed thresholds
    - Stricter volume confirmation in trending markets

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

name = "trend_momentum_v3_adx"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for better risk-adjusted returns

# EMA periods for trend detection
EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 50
EMA_MAJOR = 200

# ADX configuration for trend strength
ADX_PERIOD = 14
ADX_THRESHOLD = 25  # Minimum ADX for trending market
ADX_STRONG = 40  # Strong trend threshold

# RSI configuration with directional scoring
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_BULLISH_ZONE = 50  # Above this = bullish momentum
RSI_BEARISH_ZONE = 50  # Below this = bearish momentum

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_PERCENTILE_THRESHOLD = 0.5  # Volume must be in top 50%

# Trend scoring weights
WEIGHT_FAST = 0.4
WEIGHT_MEDIUM = 0.35
WEIGHT_SLOW = 0.25

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.010  # Target hourly volatility
VOLATILITY_MIN_PERCENTILE = 0.15  # Bottom 15% = too low
VOLATILITY_MAX_PERCENTILE = 0.85  # Top 15% = too high

# Pullback configuration
PULLBACK_LOOKBACK = 5
PULLBACK_THRESHOLD = 0.003  # 0.3% pullback from recent high/low

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


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average Directional Index using only past data.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ADX period
    
    Returns:
        Array of ADX values (0-100)
    """
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range, +DM, -DM
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM
    tr_series = pd.Series(tr)
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    
    atr = tr_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    smoothed_plus_dm = plus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    smoothed_minus_dm = minus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate +DI, -DI
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * smoothed_plus_dm[i] / atr[i]
            minus_di[i] = 100 * smoothed_minus_dm[i] / atr[i]
    
    # Calculate DX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX (smoothed DX)
    dx_series = pd.Series(dx)
    adx_series = dx_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    adx = np.nan_to_num(adx_series.values, nan=0.0)
    
    return adx


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


def calculate_volatility_percentile(atr_pct: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate ATR percentile rank using rolling window.
    Only uses past volatility data (no look-ahead).
    
    Args:
        atr_pct: Array of ATR percentage values
        lookback: Rolling window for percentile calculation
    
    Returns:
        Array of volatility percentile ranks (0-1)
    """
    n = len(atr_pct)
    vol_pct = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return vol_pct
    
    atr_series = pd.Series(atr_pct)
    
    for i in range(lookback, n):
        window = atr_series.iloc[i-lookback:i]
        rank = (window < atr_pct[i]).sum() / lookback
        vol_pct[i] = rank
    
    return vol_pct


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
    trend_score = np.clip(trend_score / 0.01, -1.0, 1.0)
    
    return trend_score


def calculate_rsi_directional_score(rsi: float, trend_direction: int) -> float:
    """
    Calculate RSI momentum score with directional bias.
    
    Args:
        rsi: Current RSI value
        trend_direction: +1 for uptrend, -1 for downtrend
    
    Returns:
        RSI score in range [0, 1]
    """
    if trend_direction > 0:
        # Bullish trend - favor higher RSI
        if rsi < RSI_OVERSOLD:
            return 0.1  # Too weak
        elif rsi < RSI_BEARISH_ZONE:
            return 0.3  # Weak momentum
        elif rsi < RSI_BULLISH_ZONE:
            return 0.5  # Neutral
        elif rsi < RSI_OVERBOUGHT:
            return 0.8  # Strong momentum
        else:
            return 0.6  # Overbought but trending
    else:
        # Bearish trend - favor lower RSI
        if rsi > RSI_OVERBOUGHT:
            return 0.1  # Too strong
        elif rsi > RSI_BULLISH_ZONE:
            return 0.3  # Weak momentum
        elif rsi > RSI_BEARISH_ZONE:
            return 0.5  # Neutral
        elif rsi > RSI_OVERSOLD:
            return 0.8  # Strong momentum
        else:
            return 0.6  # Oversold but trending


def detect_pullback(close: np.ndarray, high: np.ndarray, low: np.ndarray, 
                    i: int, lookback: int, threshold: float, direction: int) -> bool:
    """
    Detect if price is pulling back in the trend direction.
    Only uses past data (no look-ahead).
    
    Args:
        close: Array of close prices
        high: Array of high prices
        low: Array of low prices
        i: Current index
        lookback: Lookback period for recent high/low
        threshold: Minimum pullback percentage
        direction: +1 for long pullback, -1 for short pullback
    
    Returns:
        True if pullback detected
    """
    if i < lookback + 1:
        return False
    
    if direction > 0:
        # Long: looking for pullback from recent high
        recent_high = np.max(high[i-lookback:i])
        pullback_pct = (recent_high - close[i]) / recent_high
        return pullback_pct >= threshold
    else:
        # Short: looking for pullback from recent low
        recent_low = np.min(low[i-lookback:i])
        pullback_pct = (close[i] - recent_low) / recent_low
        return pullback_pct >= threshold


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum V3 Strategy with ADX Filter and Pullback Entries.
    
    Signal Logic:
    1. ADX filter to identify trending markets (ADX >= 25)
    2. Weighted trend score from EMA stack alignment
    3. Directional RSI scoring based on trend direction
    4. Pullback detection for better entry timing
    5. Volume percentile ranking for confirmation
    6. Adaptive volatility-based position sizing
    7. Signal smoothing to reduce whipsaws
    
    Entry Conditions:
    - LONG: ADX >= 25 + Positive trend + RSI bullish score + Pullback or breakout
    - SHORT: ADX >= 25 + Negative trend + RSI bearish score + Pullback or breakout
    
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
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    volume_pct = calculate_volume_percentile(volume, VOLUME_LOOKBACK)
    
    # Calculate ATR percentage for volatility analysis
    atr_pct = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if close[i] > 0:
            atr_pct[i] = atr[i] / close[i]
    
    # Calculate volatility percentile for adaptive thresholds
    vol_percentile = calculate_volatility_percentile(atr_pct, 50)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1,
        VOLUME_LOOKBACK,
        50  # For volatility percentile
    )
    
    # Track previous signal for smoothing
    prev_signal = 0.0
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check ADX filter - only trade in trending markets
        if adx[i] < ADX_THRESHOLD:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check volatility regime using adaptive percentiles
        if vol_percentile[i] < VOLATILITY_MIN_PERCENTILE or vol_percentile[i] > VOLATILITY_MAX_PERCENTILE:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate weighted trend score
        trend_score = calculate_trend_score(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i]
        )
        
        # Skip weak trends
        if abs(trend_score) < 0.15:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Determine trend direction
        trend_direction = 1 if trend_score > 0 else -1
        
        # Calculate directional RSI momentum score
        rsi_score = calculate_rsi_directional_score(rsi[i], trend_direction)
        
        # Skip if RSI momentum is too weak
        if rsi_score < 0.3:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume_pct[i] >= VOLUME_PERCENTILE_THRESHOLD
        
        # Pullback detection for better entry timing
        pullback_detected = detect_pullback(
            close, high, low, i, PULLBACK_LOOKBACK, 
            PULLBACK_THRESHOLD, trend_direction
        )
        
        # Determine signal direction and base magnitude
        if trend_direction > 0:
            # LONG bias
            base_signal = trend_score * rsi_score
        else:
            # SHORT bias
            base_signal = trend_score * rsi_score
        
        # Apply pullback bonus (better risk/reward on pullbacks)
        if pullback_detected:
            base_signal *= 1.2
        
        # Apply volume confirmation boost
        if volume_confirmed:
            base_signal *= 1.1
        
        # ADX strength bonus
        if adx[i] >= ADX_STRONG:
            base_signal *= 1.15
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct[i], 0.001)
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
    
    return signals