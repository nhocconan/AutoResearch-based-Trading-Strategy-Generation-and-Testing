#!/usr/bin/env python3
"""
strategy.py - Trend Momentum V3 with Funding Awareness
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #007/#002 success, simplifying and improving:
    - Cleaner triple-EMA trend alignment (9/21/50)
    - RSI momentum with asymmetric thresholds
    - Optional funding rate sentiment filter (if available)
    - Volatility-regime aware position sizing
    - Reduced signal smoothing for better responsiveness
    
    Key improvements over V2:
    - Simpler EMA stack (3 vs 4 EMAs)
    - Asymmetric RSI thresholds (longer bias in crypto)
    - Funding rate as sentiment filter when available
    - More conservative volatility scaling
    - Cleaner entry/exit logic

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
leverage = 2.0  # Conservative leverage for risk-adjusted returns

# EMA periods for trend detection (simplified stack)
EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 50

# RSI configuration with asymmetric thresholds (crypto bias long)
RSI_PERIOD = 14
RSI_LONG_ENTRY = 45  # Lower threshold for longs (crypto uptrend bias)
RSI_SHORT_ENTRY = 65  # Higher threshold for shorts
RSI_EXIT = 50  # Neutral exit zone

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_THRESHOLD = 0.5  # Volume must be above median

# Trend scoring
TREND_MIN_SCORE = 0.20  # Minimum trend strength to trade

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.008  # Target hourly volatility
VOLATILITY_MIN = 0.0015
VOLATILITY_MAX = 0.030

# Funding rate configuration (if available)
FUNDING_EXTREME = 0.0005  # 0.05% per 8h = extreme
FUNDING_IMPACT = 0.3  # How much funding affects signal

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.5  # Moderate smoothing


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


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio vs rolling median using only past data.
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for median calculation
    
    Returns:
        Array of volume ratios (current / median)
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    
    for i in range(lookback, n):
        window = volume_series.iloc[i-lookback:i]
        median_vol = window.median()
        if median_vol > 0:
            volume_ratio[i] = volume[i] / median_vol
        else:
            volume_ratio[i] = 1.0
    
    return volume_ratio


def calculate_trend_score(close: float, ema_fast: float, ema_medium: float, 
                          ema_slow: float) -> float:
    """
    Calculate trend score based on triple-EMA alignment.
    
    Args:
        close: Current close price
        ema_fast: Fast EMA value
        ema_medium: Medium EMA value
        ema_slow: Slow EMA value
    
    Returns:
        Trend score in range [-1, 1]
    """
    if close <= 0 or ema_slow <= 0:
        return 0.0
    
    # Check EMA stack alignment
    bullish_alignment = (ema_fast > ema_medium) and (ema_medium > ema_slow)
    bearish_alignment = (ema_fast < ema_medium) and (ema_medium < ema_slow)
    
    if not bullish_alignment and not bearish_alignment:
        # Mixed alignment - calculate based on price position
        mid_ema = (ema_fast + ema_medium + ema_slow) / 3.0
        score = (close - mid_ema) / close
        return np.clip(score * 10.0, -1.0, 1.0)
    
    # Calculate EMA spacing quality
    if bullish_alignment:
        spacing1 = (ema_fast - ema_medium) / ema_medium
        spacing2 = (ema_medium - ema_slow) / ema_slow
        base_score = 0.5 + 0.25 * np.tanh(spacing1 * 100) + 0.25 * np.tanh(spacing2 * 100)
        return base_score
    else:
        spacing1 = (ema_medium - ema_fast) / ema_fast
        spacing2 = (ema_slow - ema_medium) / ema_medium
        base_score = -0.5 - 0.25 * np.tanh(spacing1 * 100) - 0.25 * np.tanh(spacing2 * 100)
        return base_score


def calculate_rsi_signal(rsi: float, direction: int) -> float:
    """
    Calculate RSI momentum signal based on direction.
    Asymmetric thresholds favor longs in crypto.
    
    Args:
        rsi: Current RSI value
        direction: 1 for long, -1 for short
    
    Returns:
        RSI signal multiplier (0-1)
    """
    if direction > 0:
        # Long bias - more lenient entry
        if rsi < RSI_LONG_ENTRY:
            return 0.3  # Weak momentum
        elif rsi < RSI_EXIT:
            return 0.6  # Building momentum
        elif rsi < RSI_SHORT_ENTRY:
            return 0.9  # Strong momentum
        else:
            return 0.5  # Overbought caution
    else:
        # Short bias - stricter entry
        if rsi > RSI_SHORT_ENTRY:
            return 0.3  # Weak for short
        elif rsi > RSI_EXIT:
            return 0.6  # Building downside
        elif rsi > RSI_LONG_ENTRY:
            return 0.9  # Strong downside momentum
        else:
            return 0.5  # Oversold caution


def get_funding_rate(prices: pd.DataFrame, index: int) -> float:
    """
    Get funding rate if available in prices DataFrame.
    Returns 0.0 if not available.
    
    Args:
        prices: DataFrame with price data
        index: Current bar index
    
    Returns:
        Funding rate or 0.0
    """
    try:
        if 'funding_rate' in prices.columns:
            fr = prices['funding_rate'].iloc[index]
            if pd.isna(fr):
                return 0.0
            return float(fr)
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return 0.0


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum V3 Strategy with Funding Awareness.
    
    Signal Logic:
    1. Triple-EMA trend alignment scoring
    2. RSI momentum with asymmetric thresholds
    3. Volume confirmation (above median)
    4. Funding rate sentiment filter (if available)
    5. Volatility-based position sizing
    
    Entry Conditions:
    - LONG: Bullish EMA stack + RSI > 45 + volume confirmed
    - SHORT: Bearish EMA stack + RSI < 65 + volume confirmed
    
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
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK
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
        
        # Check volatility regime (avoid extreme volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate trend score
        trend_score = calculate_trend_score(
            close[i], ema_fast[i], ema_medium[i], ema_slow[i]
        )
        
        # Skip weak trends
        if abs(trend_score) < TREND_MIN_SCORE:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Determine direction
        direction = 1 if trend_score > 0 else -1
        
        # Calculate RSI signal multiplier
        rsi_signal = calculate_rsi_signal(rsi[i], direction)
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_THRESHOLD
        volume_multiplier = 1.0 if volume_confirmed else 0.7
        
        # Base signal calculation
        base_signal = trend_score * rsi_signal * volume_multiplier
        
        # Funding rate filter (if available)
        funding_rate = get_funding_rate(prices, i)
        if abs(funding_rate) > FUNDING_EXTREME:
            # Extreme funding - reduce position in direction of funding
            # (crowded trade warning)
            if direction > 0 and funding_rate > FUNDING_EXTREME:
                base_signal *= (1.0 - FUNDING_IMPACT)
            elif direction < 0 and funding_rate < -FUNDING_EXTREME:
                base_signal *= (1.0 - FUNDING_IMPACT)
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.4, 1.8)
        
        raw_signal = base_signal * vol_factor
        
        # Apply exponential smoothing to reduce whipsaws
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        prev_signal = smoothed_signal
        
        # Apply minimum threshold
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals