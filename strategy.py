#!/usr/bin/env python3
"""
strategy.py - Funding Mean Reversion V13
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

Strategy Hypothesis:
    Funding rate mean reversion with trend confirmation:
    - Primary signal: Extreme funding rates indicate crowded positions
    - Extreme positive funding → short bias (crowded longs will unwind)
    - Extreme negative funding → long bias (crowded shorts will unwind)
    - Trend confirmation: Only take mean reversion trades WITH the major trend
    - Entry timing: RSI divergence for better entry points
    - Volatility filter: Avoid trading during extreme volatility spikes
    
    Why this might work better:
    - Previous trend-following strategies failed (all negative returns)
    - Funding extremes are reliable mean reversion signals in crypto
    - Trend confirmation prevents fighting major moves
    - Simpler logic = less signal dilution and lag
    
    Key changes from v12:
    - Funding is PRIMARY signal, not overlay
    - Removed excessive smoothing/hysteresis causing lag
    - Simpler EMA configuration (12/26 vs 21/55)
    - Only filter out extreme against-trend funding, don't overlay
    - Reduced number of conflicting conditions

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

name = "funding_mean_reversion_v13"
timeframe = "1h"
leverage = 2.0  # Conservative given historical performance

# EMA configuration for trend confirmation
EMA_FAST = 12
EMA_SLOW = 26
EMA_MAJOR = 200

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Funding rate configuration (PRIMARY SIGNAL)
FUNDING_EXTREME_LONG = 0.0010   # 0.10% per 8hr - extreme positive
FUNDING_EXTREME_SHORT = -0.0010 # -0.10% per 8hr - extreme negative
FUNDING_MODERATE_LONG = 0.0004  # 0.04% per 8hr - moderate positive
FUNDING_MODERATE_SHORT = -0.0004 # -0.04% per 8hr - moderate negative
FUNDING_LOOKBACK = 96  # 4 days of hourly data for percentile

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_MAX = 0.08  # Skip if ATR% > 8% (extreme volatility)

# Signal configuration
MIN_SIGNAL = 0.25  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.4  # EMA smoothing factor for signals


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
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


def calculate_funding_percentile(funding_rate: np.ndarray, lookback: int = 96) -> np.ndarray:
    """
    Calculate rolling percentile rank of funding rate.
    Returns value from 0 to 1 (higher = more extreme positive).
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    percentile = np.full(n, 0.5, dtype=np.float64)
    
    if n < lookback:
        return percentile
    
    funding_series = pd.Series(funding_rate)
    
    for i in range(lookback - 1, n):
        window = funding_rate[i - lookback + 1:i + 1]
        if len(window) > 0 and not np.all(window == window[0]):
            rank = np.sum(window < funding_rate[i]) / len(window)
            percentile[i] = rank
    
    return percentile


def calculate_funding_signal(funding_rate: np.ndarray,
                             funding_pct: np.ndarray) -> np.ndarray:
    """
    Calculate funding rate mean reversion signal.
    Extreme positive funding → short bias (negative signal)
    Extreme negative funding → long bias (positive signal)
    Returns value in [-1, 1].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        fr = funding_rate[i]
        pct = funding_pct[i]
        
        # Strong signals based on absolute funding rate
        if fr >= FUNDING_EXTREME_LONG:
            # Extreme positive funding - strong short signal
            signal[i] = -1.0 * min(1.5, fr / FUNDING_EXTREME_LONG)
        elif fr <= FUNDING_EXTREME_SHORT:
            # Extreme negative funding - strong long signal
            signal[i] = 1.0 * min(1.5, abs(fr) / abs(FUNDING_EXTREME_SHORT))
        elif fr >= FUNDING_MODERATE_LONG:
            # Moderate positive funding - mild short signal
            signal[i] = -0.5 * (fr / FUNDING_MODERATE_LONG)
        elif fr <= FUNDING_MODERATE_SHORT:
            # Moderate negative funding - mild long signal
            signal[i] = 0.5 * (abs(fr) / abs(FUNDING_MODERATE_SHORT))
        # Else: neutral funding, no signal
        
    return np.clip(signal, -1.0, 1.0)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Funding Mean Reversion V13 Strategy.
    
    Signal Logic:
    1. Calculate funding rate signal (PRIMARY)
    2. Confirm with major trend direction (200 EMA)
    3. Filter with RSI for entry timing
    4. Skip during extreme volatility
    5. Smooth signals slightly to reduce noise
    6. Apply minimum magnitude filter
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, funding_rate, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract price data with error handling
    try:
        close = prices["close"].values.astype(np.float64)
        high = prices["high"].values.astype(np.float64)
        low = prices["low"].values.astype(np.float64)
        
        try:
            funding_rate = prices["funding_rate"].values.astype(np.float64)
            funding_rate = np.nan_to_num(funding_rate, nan=0.0)
        except (KeyError, TypeError, ValueError):
            funding_rate = np.zeros(n, dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Clean data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    
    # Fix invalid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators (all use only past data)
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    funding_pct = calculate_funding_percentile(funding_rate, FUNDING_LOOKBACK)
    funding_signal = calculate_funding_signal(funding_rate, funding_pct)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        FUNDING_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volatility filter - skip extreme volatility
        atr_pct = atr[i] / close[i]
        if atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Get raw funding signal
        raw_funding_sig = funding_signal[i]
        
        # If no funding signal, skip
        if abs(raw_funding_sig) < 0.1:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Determine signal direction
        signal_direction = np.sign(raw_funding_sig)
        
        # Trend confirmation filter
        # For LONG signals (negative funding): price should be >= 200 EMA or not far below
        # For SHORT signals (positive funding): price should be <= 200 EMA or not far above
        price_vs_major = (close[i] - ema_major[i]) / close[i]
        
        if signal_direction > 0:  # Long signal from negative funding
            # Allow longs if price is at or above major EMA, or slightly below
            if price_vs_major < -0.03:  # More than 3% below major EMA
                raw_funding_sig = 0.0
        elif signal_direction < 0:  # Short signal from positive funding
            # Allow shorts if price is at or below major EMA, or slightly above
            if price_vs_major > 0.03:  # More than 3% above major EMA
                raw_funding_sig = 0.0
        
        # Skip if filtered out
        if raw_funding_sig == 0.0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # RSI entry timing filter
        if signal_direction > 0:  # Long
            # Avoid entering when RSI is overbought
            if rsi[i] > RSI_OVERBOUGHT:
                raw_funding_sig *= 0.5
        elif signal_direction < 0:  # Short
            # Avoid entering when RSI is oversold
            if rsi[i] < RSI_OVERSOLD:
                raw_funding_sig *= 0.5
        
        # EMA crossover confirmation (optional boost)
        ema_trend = np.sign(ema_fast[i] - ema_slow[i])
        if ema_trend == signal_direction:
            # Trend aligns with funding signal - boost confidence
            raw_funding_sig *= 1.2
        elif ema_trend == -signal_direction:
            # Trend opposes funding signal - reduce confidence
            raw_funding_sig *= 0.7
        
        # Apply signal smoothing
        smoothed_signal = (SIGNAL_SMOOTHING * prev_signal + 
                          (1.0 - SIGNAL_SMOOTHING) * raw_funding_sig)
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals