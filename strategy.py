#!/usr/bin/env python3
"""
strategy.py - Adaptive Trend Momentum V3 with Funding Awareness
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #010 success (Sharpe=0.330, Return=+40.7%), improving:
    - Funding rate awareness for overcrowded position detection
    - Adaptive RSI scoring based on trend direction
    - Volume momentum confirmation (not just percentile)
    - Volatility-adaptive signal smoothing
    - Better trend strength measurement with EMA slope
    
    Key improvements over #010:
    - Funding rate filter (mean reversion on extremes)
    - RSI scoring adapts to trend direction (bullish vs bearish)
    - Volume momentum (rate of change) + percentile
    - Smoothing factor adjusts with volatility
    - EMA slope confirmation for trend validity

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

name = "adaptive_trend_momentum_v3"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for risk-adjusted returns

# EMA periods for trend detection
EMA_FAST = 8
EMA_MEDIUM = 21
EMA_SLOW = 55
EMA_MAJOR = 200

# RSI configuration with adaptive scoring
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_NEUTRAL_LOW = 45
RSI_NEUTRAL_HIGH = 55

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_PERCENTILE_THRESHOLD = 0.55
VOLUME_MOMENTUM_PERIOD = 5

# Trend scoring weights
WEIGHT_FAST = 0.35
WEIGHT_MEDIUM = 0.35
WEIGHT_SLOW = 0.30

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012  # Target hourly volatility
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.040

# Funding rate configuration
FUNDING_EXTREME_THRESHOLD = 0.0005  # 0.05% per 8h is extreme
FUNDING_COOLDOWN_BARS = 12  # Bars to wait after extreme funding

# Signal configuration
MIN_SIGNAL = 0.12
MAX_SIGNAL = 0.75
SMOOTHING_BASE = 0.65  # Base exponential smoothing factor


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


def calculate_ema_slope(ema: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Calculate EMA slope (rate of change) using only past data.
    
    Args:
        ema: Array of EMA values
        lookback: Period for slope calculation
    
    Returns:
        Array of slope values (normalized by price)
    """
    n = len(ema)
    slope = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return slope
    
    for i in range(lookback, n):
        if ema[i-lookback] > 0:
            slope[i] = (ema[i] - ema[i-lookback]) / ema[i-lookback]
    
    return slope


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


def calculate_volume_momentum(volume: np.ndarray, period: int = 5) -> np.ndarray:
    """
    Calculate volume momentum (rate of change) using only past data.
    
    Args:
        volume: Array of volume values
        period: Lookback period for momentum calculation
    
    Returns:
        Array of volume momentum values
    """
    n = len(volume)
    vol_momentum = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return vol_momentum
    
    for i in range(period, n):
        if volume[i-period] > 0:
            vol_momentum[i] = (volume[i] - volume[i-period]) / volume[i-period]
    
    return vol_momentum


def calculate_trend_score(close: float, ema_fast: float, ema_medium: float, 
                          ema_slow: float, ema_major: float,
                          ema_fast_slope: float, ema_medium_slope: float) -> float:
    """
    Calculate weighted trend score based on EMA stack alignment and slope.
    
    Args:
        close: Current close price
        ema_fast: Fast EMA value
        ema_medium: Medium EMA value
        ema_slow: Slow EMA value
        ema_major: Major EMA value
        ema_fast_slope: Fast EMA slope
        ema_medium_slope: Medium EMA slope
    
    Returns:
        Trend score in range [-1, 1]
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Calculate individual trend components (normalized by price)
    fast_score = (ema_fast - ema_medium) / close
    medium_score = (ema_medium - ema_slow) / close
    slow_score = (ema_slow - ema_major) / close
    
    # Weight and combine alignment scores
    alignment_score = (
        WEIGHT_FAST * fast_score +
        WEIGHT_MEDIUM * medium_score +
        WEIGHT_SLOW * slow_score
    )
    
    # Slope confirmation (adds conviction if EMAs are moving in trend direction)
    slope_confirmation = np.sign(ema_fast_slope) * abs(ema_fast_slope) * 10 + \
                         np.sign(ema_medium_slope) * abs(ema_medium_slope) * 5
    
    # Combine alignment and slope
    trend_score = alignment_score + slope_confirmation
    
    # Major trend filter (amplifies signal in direction of major trend)
    major_direction = np.sign(close - ema_major)
    trend_score = trend_score * (1.0 + 0.3 * major_direction)
    
    # Normalize to [-1, 1] range
    trend_score = np.clip(trend_score / 0.015, -1.0, 1.0)
    
    return trend_score


def calculate_rsi_score(rsi: float, trend_direction: int) -> float:
    """
    Calculate adaptive RSI momentum score based on trend direction.
    
    Args:
        rsi: Current RSI value
        trend_direction: +1 for long bias, -1 for short bias
    
    Returns:
        RSI score in range [0, 1]
    """
    if trend_direction > 0:
        # Long bias: prefer RSI in 45-70 range (momentum without overbought)
        if rsi < RSI_OVERSOLD:
            return 0.2  # Too weak
        elif rsi < RSI_NEUTRAL_LOW:
            return 0.5  # Recovering
        elif rsi < RSI_NEUTRAL_HIGH:
            return 0.7  # Good momentum
        elif rsi < RSI_OVERBOUGHT:
            return 0.9  # Strong momentum
        else:
            return 0.4  # Overbought, caution
    else:
        # Short bias: prefer RSI in 30-55 range (weakness without oversold)
        if rsi > RSI_OVERBOUGHT:
            return 0.2  # Too strong
        elif rsi > RSI_NEUTRAL_HIGH:
            return 0.5  # Weakening
        elif rsi > RSI_NEUTRAL_LOW:
            return 0.7  # Good weakness
        elif rsi > RSI_OVERSOLD:
            return 0.9  # Strong weakness
        else:
            return 0.4  # Oversold, caution


def get_funding_signal(prices: pd.DataFrame, i: int) -> float:
    """
    Get funding rate signal if available in prices DataFrame.
    Returns 0 if funding data not available.
    
    Args:
        prices: DataFrame with prices
        i: Current index
    
    Returns:
        Funding signal (-1 to 1, 0 if not available)
    """
    try:
        if 'funding_rate' in prices.columns:
            funding = prices['funding_rate'].iloc[i]
            if abs(funding) > FUNDING_EXTREME_THRESHOLD:
                # Extreme funding → mean reversion signal
                return -np.sign(funding) * 0.5
        return 0.0
    except (KeyError, TypeError, IndexError):
        return 0.0


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Trend Momentum V3 Strategy with Funding Awareness.
    
    Signal Logic:
    1. Weighted trend score from EMA stack + slope confirmation
    2. Adaptive RSI scoring based on trend direction
    3. Volume momentum + percentile confirmation
    4. Funding rate awareness for overcrowded positions
    5. Volatility-adaptive signal smoothing
    
    Entry Conditions:
    - LONG: Positive trend + RSI confirmation + volume + no extreme positive funding
    - SHORT: Negative trend + RSI confirmation + volume + no extreme negative funding
    
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
    
    # Calculate EMA slopes
    ema_fast_slope = calculate_ema_slope(ema_fast, 5)
    ema_medium_slope = calculate_ema_slope(ema_medium, 5)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_pct = calculate_volume_percentile(volume, VOLUME_LOOKBACK)
    vol_momentum = calculate_volume_momentum(volume, VOLUME_MOMENTUM_PERIOD)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        VOLUME_MOMENTUM_PERIOD
    )
    
    # Track previous signal for smoothing
    prev_signal = 0.0
    funding_cooldown = 0
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check volatility regime (avoid extreme volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate weighted trend score with slope confirmation
        trend_score = calculate_trend_score(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i],
            ema_fast_slope[i], ema_medium_slope[i]
        )
        
        # Skip weak trends
        if abs(trend_score) < 0.18:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Determine trend direction
        trend_direction = np.sign(trend_score)
        
        # Calculate adaptive RSI momentum score
        rsi_score = calculate_rsi_score(rsi[i], trend_direction)
        
        # Volume confirmation (both percentile and momentum)
        volume_confirmed = (
            volume_pct[i] >= VOLUME_PERCENTILE_THRESHOLD and
            vol_momentum[i] > -0.3  # Not collapsing volume
        )
        
        # Funding rate check
        funding_signal = get_funding_signal(prices, i)
        
        # Skip if funding indicates overcrowded position against us
        if funding_cooldown > 0:
            funding_cooldown -= 1
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        if abs(funding_signal) > 0.3:
            # Extreme funding detected, set cooldown
            funding_cooldown = FUNDING_COOLDOWN_BARS
            # Only take trades aligned with funding mean reversion
            if np.sign(funding_signal) != trend_direction:
                signals[i] = 0.0
                prev_signal = 0.0
                continue
        
        # Determine base signal magnitude
        if trend_direction > 0:
            # LONG bias
            if rsi_score < 0.4:
                base_signal = 0.0
            else:
                base_signal = trend_score * rsi_score
        else:
            # SHORT bias
            if rsi_score < 0.4:
                base_signal = 0.0
            else:
                base_signal = trend_score * rsi_score
        
        # Apply volume confirmation boost
        if volume_confirmed:
            base_signal *= 1.20
        
        # Apply funding adjustment (reduce position if funding against us)
        if funding_signal != 0:
            base_signal *= (1.0 - abs(funding_signal) * 0.5)
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.4, 2.5)
        
        raw_signal = base_signal * vol_factor
        
        # Adaptive smoothing (more smoothing in high volatility)
        adaptive_smoothing = SMOOTHING_BASE + 0.2 * min(atr_pct / VOLATILITY_TARGET, 1.0)
        adaptive_smoothing = np.clip(adaptive_smoothing, 0.5, 0.85)
        
        smoothed_signal = adaptive_smoothing * prev_signal + (1.0 - adaptive_smoothing) * raw_signal
        prev_signal = smoothed_signal
        
        # Apply thresholds
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals