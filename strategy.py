#!/usr/bin/env python3
"""
strategy.py - Multi-Timeframe Trend with Momentum Confirmation
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #004 success (Sharpe=0.208), adding multi-timeframe trend filter.
    - Primary signal: EMA stack alignment (9/21/50/200)
    - Higher timeframe filter: 200 EMA defines major trend direction
    - Momentum: RSI with dynamic thresholds based on trend strength
    - Volume: Breakout confirmation with volume spike
    - Volatility: ATR-based position sizing with regime adjustment
    
    Key improvements over #004:
    - 200 EMA major trend filter (avoid counter-trend trades)
    - Dynamic RSI thresholds (wider in strong trends)
    - Volume spike detection for breakouts
    - Better volatility regime detection
    - Cleaner signal scaling

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

name = "multi_tf_trend_momentum"
timeframe = "1h"
leverage = 2.5  # Conservative leverage given crypto volatility

# EMA periods for multi-timeframe trend detection
EMA_FAST = 9                    # Short-term momentum
EMA_MEDIUM = 21                 # Medium-term trend
EMA_SLOW = 50                   # Long-term trend
EMA_MAJOR = 200                 # Major trend filter (multi-timeframe proxy)

# RSI configuration with dynamic thresholds
RSI_PERIOD = 14
RSI_BASE_LONG_MIN = 45          # Base minimum RSI for longs
RSI_BASE_LONG_MAX = 75          # Base maximum RSI for longs
RSI_BASE_SHORT_MIN = 25         # Base minimum RSI for shorts
RSI_BASE_SHORT_MAX = 55         # Base maximum RSI for shorts

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_THRESHOLD = 1.5    # Volume must be 1.5x average for breakout confirmation
VOLUME_BASE_THRESHOLD = 1.0     # Base volume threshold

# Trend strength thresholds
TREND_STRENGTH_MIN = 0.001      # Minimum EMA spread ratio
TREND_ALIGNMENT_MIN = 0.0005    # Minimum alignment between EMAs

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012       # Target hourly volatility
VOLATILITY_MIN = 0.003          # Minimum volatility to trade
VOLATILITY_MAX = 0.040          # Maximum volatility (avoid extreme moves)

# Signal configuration
MIN_SIGNAL = 0.12               # Minimum signal magnitude to trade
MAX_SIGNAL = 0.85               # Maximum signal magnitude


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


def calculate_volatility_regime(atr: np.ndarray, close: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate volatility regime (0=low, 1=normal, 2=high).
    Uses rolling percentile of ATR/close ratio.
    
    Args:
        atr: Array of ATR values
        close: Array of close prices
        lookback: Rolling window for regime calculation
    
    Returns:
        Array of volatility regime values (0, 1, or 2)
    """
    n = len(close)
    regime = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return regime
    
    atr_pct = atr / np.where(close > 0, close, 1.0)
    atr_pct_series = pd.Series(atr_pct)
    
    for i in range(lookback, n):
        window = atr_pct_series.iloc[i-lookback:i]
        percentile = (window <= atr_pct[i]).mean()
        
        if percentile < 0.3:
            regime[i] = 0.0  # Low volatility
        elif percentile < 0.7:
            regime[i] = 1.0  # Normal volatility
        else:
            regime[i] = 2.0  # High volatility
    
    return regime


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Multi-Timeframe Trend Following Strategy with Momentum Confirmation.
    
    Signal Logic:
    1. Major trend: Price relative to 200 EMA defines overall direction
    2. Trend alignment: 9 > 21 > 50 > 200 (bullish) or reverse (bearish)
    3. Momentum filter: RSI in reasonable range (dynamic thresholds)
    4. Volume confirmation: Volume spike for breakout validation
    5. Volatility scaling: ATR-based position sizing with regime adjustment
    
    Entry Conditions:
    - LONG: Price > EMA200 + EMA9>21>50 + RSI 45-75 + volume confirmation
    - SHORT: Price < EMA200 + EMA9<21<50 + RSI 25-55 + volume confirmation
    
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
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    vol_regime = calculate_volatility_regime(atr, close, 50)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        50  # For volatility regime
    )
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Check volatility regime (avoid extreme volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            continue
        
        # Major trend filter (200 EMA)
        price_vs_major = (close[i] - ema_major[i]) / close[i]
        major_trend_bullish = price_vs_major > TREND_ALIGNMENT_MIN
        major_trend_bearish = price_vs_major < -TREND_ALIGNMENT_MIN
        
        # EMA stack alignment
        ema_stack_bullish = (ema_fast[i] > ema_medium[i] > ema_slow[i] > ema_major[i])
        ema_stack_bearish = (ema_fast[i] < ema_medium[i] < ema_slow[i] < ema_major[i])
        
        # Calculate trend strength
        trend_strength_fast = abs(ema_fast[i] - ema_medium[i]) / close[i]
        trend_strength_medium = abs(ema_medium[i] - ema_slow[i]) / close[i]
        trend_strength_slow = abs(ema_slow[i] - ema_major[i]) / close[i]
        trend_strength = min(trend_strength_fast, trend_strength_medium, trend_strength_slow)
        
        # Filter: trend must be strong enough and aligned
        if trend_strength < TREND_STRENGTH_MIN:
            signals[i] = 0.0
            continue
        
        # Dynamic RSI thresholds based on trend strength
        rsi_adjustment = min(trend_strength / 0.005, 0.5)
        rsi_long_min = RSI_BASE_LONG_MIN - rsi_adjustment * 10
        rsi_long_max = RSI_BASE_LONG_MAX + rsi_adjustment * 5
        rsi_short_min = RSI_BASE_SHORT_MIN - rsi_adjustment * 5
        rsi_short_max = RSI_BASE_SHORT_MAX + rsi_adjustment * 10
        
        # RSI momentum filter
        rsi_long_ok = rsi_long_min <= rsi[i] <= rsi_long_max
        rsi_short_ok = rsi_short_min <= rsi[i] <= rsi_short_max
        
        # Volume confirmation
        volume_base_ok = volume_ratio[i] >= VOLUME_BASE_THRESHOLD
        volume_spike = volume_ratio[i] >= VOLUME_SPIKE_THRESHOLD
        
        # Calculate signal
        raw_signal = 0.0
        signal_confidence = 0.0
        
        # LONG signal
        if major_trend_bullish and ema_stack_bullish and rsi_long_ok:
            base_confidence = 0.5
            
            # Trend strength factor
            trend_factor = min(trend_strength / 0.006, 1.0)
            base_confidence += trend_factor * 0.3
            
            # Volume boost
            if volume_spike:
                base_confidence *= 1.2
            elif volume_base_ok:
                base_confidence *= 1.05
            
            # RSI quality (prefer momentum in trending market)
            rsi_quality = 1.0
            if 50 <= rsi[i] <= 65:
                rsi_quality = 1.0
            elif rsi_long_min <= rsi[i] < 50 or 65 < rsi[i] <= rsi_long_max:
                rsi_quality = 0.85
            
            # Volatility regime adjustment
            regime_factor = 1.0
            if vol_regime[i] == 0:
                regime_factor = 1.1  # Low vol = more confidence
            elif vol_regime[i] == 2:
                regime_factor = 0.8  # High vol = less confidence
            
            signal_confidence = base_confidence * rsi_quality * regime_factor
            raw_signal = signal_confidence
        
        # SHORT signal
        elif major_trend_bearish and ema_stack_bearish and rsi_short_ok:
            base_confidence = 0.5
            
            trend_factor = min(trend_strength / 0.006, 1.0)
            base_confidence += trend_factor * 0.3
            
            if volume_spike:
                base_confidence *= 1.2
            elif volume_base_ok:
                base_confidence *= 1.05
            
            # RSI quality
            rsi_quality = 1.0
            if 35 <= rsi[i] <= 50:
                rsi_quality = 1.0
            elif rsi_short_min <= rsi[i] < 35 or 50 < rsi[i] <= rsi_short_max:
                rsi_quality = 0.85
            
            # Volatility regime adjustment
            regime_factor = 1.0
            if vol_regime[i] == 0:
                regime_factor = 1.1
            elif vol_regime[i] == 2:
                regime_factor = 0.8
            
            signal_confidence = base_confidence * rsi_quality * regime_factor
            raw_signal = -signal_confidence
        
        # Apply volatility adjustment for position sizing
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