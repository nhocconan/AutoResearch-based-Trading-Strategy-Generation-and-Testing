#!/usr/bin/env python3
"""
strategy.py - Volatility Breakout with Momentum Confirmation
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #004's success (Sharpe=0.208), this strategy focuses on:
    - Bollinger Band squeeze detection (low volatility consolidation)
    - Breakout confirmation with volume spike
    - RSI momentum for direction confirmation
    - EMA trend filter for alignment
    
    Key improvements over #004:
    - BB squeeze detection identifies consolidation before explosive moves
    - Volume spike REQUIRED for breakout (not optional)
    - More adaptive RSI thresholds based on trend strength
    - Better risk management with ATR-based position sizing
    
    Why this should work:
    - Crypto markets often consolidate before major moves
    - Volume confirms genuine breakouts vs fakeouts
    - Combines mean reversion (squeeze) with trend following (breakout)

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

name = "volatility_breakout_momentum"
timeframe = "1h"
leverage = 3.0  # Moderate leverage for breakout capture

# Bollinger Band parameters
BB_PERIOD = 20                    # Standard BB period
BB_STD_DEV = 2.0                  # Standard deviations for bands
BB_SQUEEZE_THRESHOLD = 0.015      # Bandwidth threshold for squeeze detection

# EMA parameters for trend filter
EMA_FAST = 8                      # Fast EMA for entry timing
EMA_SLOW = 21                     # Slow EMA for trend direction
EMA_CONFIRM = 50                  # Confirmation EMA

# Volume parameters
VOLUME_LOOKBACK = 20              # Lookback for volume average
VOLUME_SPIKE_THRESHOLD = 1.5      # Volume must be 1.5x average for breakout

# RSI parameters
RSI_PERIOD = 14                   # RSI calculation period
RSI_LONG_MIN = 45                 # Minimum RSI for long entries
RSI_LONG_MAX = 70                 # Maximum RSI for long entries (not overbought)
RSI_SHORT_MIN = 30                # Minimum RSI for short entries
RSI_SHORT_MAX = 55                # Maximum RSI for short entries

# ATR and risk management
ATR_PERIOD = 14                   # ATR calculation period
VOLATILITY_TARGET = 0.012         # Target hourly volatility for position sizing
MIN_SIGNAL = 0.20                 # Minimum signal magnitude to trade
MAX_SIGNAL = 0.85                 # Maximum signal magnitude

# Breakout confirmation
BREAKOUT_LOOKBACK = 5             # Lookback for breakout confirmation
PRICE_CHANGE_THRESHOLD = 0.008    # Minimum price change for breakout


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


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands using only past data.
    
    Args:
        close: Array of close prices
        period: Rolling window period
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
    rolling_mean = close_series.rolling(window=period, min_periods=period).mean()
    rolling_std = close_series.rolling(window=period, min_periods=period).std()
    
    middle = np.nan_to_num(rolling_mean.values, nan=0.0)
    std_values = np.nan_to_num(rolling_std.values, nan=0.0)
    
    upper = middle + (std_dev * std_values)
    lower = middle - (std_dev * std_values)
    
    # Bandwidth = (Upper - Lower) / Middle
    mask = middle > 0
    bandwidth[mask] = (upper[mask] - lower[mask]) / middle[mask]
    
    return upper, middle, lower, bandwidth


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
    Calculate volume ratio relative to rolling average.
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
    rolling_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean().values
    
    mask = rolling_avg > 0
    volume_ratio[mask] = volume[mask] / rolling_avg[mask]
    
    return volume_ratio


def calculate_price_momentum(close: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Calculate price momentum (percentage change over lookback period).
    Only uses past data.
    
    Args:
        close: Array of close prices
        lookback: Lookback period for momentum calculation
    
    Returns:
        Array of momentum values (percentage change)
    """
    n = len(close)
    momentum = np.zeros(n, dtype=np.float64)
    
    if n < lookback + 1:
        return momentum
    
    for i in range(lookback, n):
        if close[i - lookback] > 0:
            momentum[i] = (close[i] - close[i - lookback]) / close[i - lookback]
    
    return momentum


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Volatility Breakout Strategy with Momentum Confirmation.
    
    Signal Logic:
    1. BB Squeeze Detection: Bandwidth below threshold indicates consolidation
    2. Breakout Confirmation: Price breaks above/below BB with volume spike
    3. Trend Filter: EMA alignment confirms direction
    4. Momentum Filter: RSI in reasonable range (not overextended)
    5. Price Momentum: Recent price change confirms breakout strength
    
    Entry Conditions:
    - LONG: BB squeeze + price > upper BB + volume spike + EMA bullish + RSI 45-70
    - SHORT: BB squeeze + price < lower BB + volume spike + EMA bearish + RSI 30-55
    
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
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_confirm = calculate_ema(close, EMA_CONFIRM)
    
    bb_upper, bb_middle, bb_lower, bb_bandwidth = calculate_bollinger_bands(
        close, BB_PERIOD, BB_STD_DEV
    )
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    price_momentum = calculate_price_momentum(close, BREAKOUT_LOOKBACK)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_CONFIRM,
        BB_PERIOD,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        BREAKOUT_LOOKBACK + 1
    )
    
    # Track squeeze state for better signal quality
    in_squeeze = np.zeros(n, dtype=bool)
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0 or bb_middle[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Detect BB squeeze (low volatility consolidation)
        is_squeeze = bb_bandwidth[i] < BB_SQUEEZE_THRESHOLD
        in_squeeze[i] = is_squeeze
        
        # Check for squeeze in recent past (consolidation period)
        recent_squeeze = np.any(in_squeeze[max(0, i - BB_PERIOD):i])
        
        # Trend direction and strength
        trend_bullish = (ema_fast[i] > ema_slow[i] > ema_confirm[i])
        trend_bearish = (ema_fast[i] < ema_slow[i] < ema_confirm[i])
        
        # EMA spread for trend strength
        ema_spread = (ema_fast[i] - ema_slow[i]) / close[i]
        trend_strength = abs(ema_spread)
        
        # Volume confirmation (REQUIRED for breakout)
        volume_confirmed = volume_ratio[i] >= VOLUME_SPIKE_THRESHOLD
        
        # RSI momentum filter
        rsi_long_ok = RSI_LONG_MIN <= rsi[i] <= RSI_LONG_MAX
        rsi_short_ok = RSI_SHORT_MIN <= rsi[i] <= RSI_SHORT_MAX
        
        # Price momentum confirmation
        momentum_long = price_momentum[i] > PRICE_CHANGE_THRESHOLD
        momentum_short = price_momentum[i] < -PRICE_CHANGE_THRESHOLD
        
        # Breakout detection
        breakout_long = close[i] > bb_upper[i]
        breakout_short = close[i] < bb_lower[i]
        
        # Calculate signal
        raw_signal = 0.0
        signal_confidence = 0.0
        
        if breakout_long and trend_bullish and rsi_long_ok and volume_confirmed:
            # Long breakout signal
            base_confidence = 0.5
            
            # Boost confidence for squeeze breakout
            if recent_squeeze:
                base_confidence += 0.2
            
            # Add confidence for stronger trend
            trend_factor = min(trend_strength / 0.005, 1.0)
            base_confidence += trend_factor * 0.2
            
            # Momentum quality
            momentum_factor = min(abs(price_momentum[i]) / 0.02, 1.0)
            base_confidence += momentum_factor * 0.1
            
            # RSI quality (prefer mid-range momentum)
            rsi_quality = 1.0
            if 50 <= rsi[i] <= 65:
                rsi_quality = 1.0
            elif 45 <= rsi[i] < 50 or 65 < rsi[i] <= 70:
                rsi_quality = 0.85
            
            signal_confidence = base_confidence * rsi_quality
            raw_signal = signal_confidence
            
        elif breakout_short and trend_bearish and rsi_short_ok and volume_confirmed:
            # Short breakout signal
            base_confidence = 0.5
            
            # Boost confidence for squeeze breakout
            if recent_squeeze:
                base_confidence += 0.2
            
            trend_factor = min(trend_strength / 0.005, 1.0)
            base_confidence += trend_factor * 0.2
            
            momentum_factor = min(abs(price_momentum[i]) / 0.02, 1.0)
            base_confidence += momentum_factor * 0.1
            
            # RSI quality
            rsi_quality = 1.0
            if 35 <= rsi[i] <= 50:
                rsi_quality = 1.0
            elif 30 <= rsi[i] < 35 or 50 < rsi[i] <= 55:
                rsi_quality = 0.85
            
            signal_confidence = base_confidence * rsi_quality
            raw_signal = -signal_confidence
        
        # Apply volatility adjustment
        atr_pct = atr[i] / close[i]
        vol_factor = 1.0
        if atr_pct > 0:
            vol_factor = min(1.5, VOLATILITY_TARGET / max(atr_pct, 0.001))
        
        signal = raw_signal * vol_factor
        
        # Apply thresholds
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        signal = np.clip(signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals