#!/usr/bin/env python3
"""
strategy.py - Multi-Timeframe Trend with Volatility Filter
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified version of #008 that fixed the crash while keeping core concepts:
    - Multi-timeframe EMA stack for trend direction
    - Bollinger Band width for volatility regime detection
    - RSI filter for momentum confirmation
    - Cleaner array operations to avoid dimension errors
    - Reduced parameter count for better robustness
    
    Key fixes from #008:
    - Removed complex loops that caused array dimension errors
    - Simplified volatility regime calculation
    - Cleaner pandas-based indicator calculations
    - Better NaN handling throughout

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

name = "multi_tf_trend_vol_filter"
timeframe = "1h"
leverage = 2.0  # Conservative leverage

# EMA periods for trend detection
EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 50
EMA_MAJOR = 200

# Bollinger Band configuration
BB_PERIOD = 20
BB_STD_DEV = 2.0
BB_WIDTH_LOW = 0.02   # Squeeze threshold
BB_WIDTH_HIGH = 0.08  # Expansion threshold

# RSI configuration
RSI_PERIOD = 14
RSI_LONG_MIN = 40
RSI_LONG_MAX = 70
RSI_SHORT_MIN = 30
RSI_SHORT_MAX = 60

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_THRESHOLD = 1.2

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.80
SIGNAL_SMOOTHING = 3

# Volatility filter
ATR_PERIOD = 14
ATR_MIN_PCT = 0.003
ATR_MAX_PCT = 0.035


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Multi-Timeframe Trend Strategy with Volatility Filter.
    
    Signal Logic:
    1. Major trend: Price relative to 200 EMA
    2. Trend alignment: EMA stack (9/21/50/200)
    3. Volatility filter: ATR percentage in reasonable range
    4. BB width: Avoid extreme compression or expansion
    5. RSI filter: Momentum confirmation
    6. Volume: Basic confirmation
    
    Entry Conditions:
    - LONG: Price > EMA200 + EMA stack bullish + RSI ok + BB ok
    - SHORT: Price < EMA200 + EMA stack bearish + RSI ok + BB ok
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract required columns with safety checks
    try:
        close = prices["close"].astype(np.float64).values
        high = prices["high"].astype(np.float64).values
        low = prices["low"].astype(np.float64).values
        volume = prices["volume"].astype(np.float64).values
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Handle NaN values
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Ensure valid prices (avoid division by zero)
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Create pandas series for easier calculations
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    volume_series = pd.Series(volume)
    
    # Calculate EMAs
    ema_fast = close_series.ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_medium = close_series.ewm(span=EMA_MEDIUM, adjust=False, min_periods=EMA_MEDIUM).mean().values
    ema_slow = close_series.ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    ema_major = close_series.ewm(span=EMA_MAJOR, adjust=False, min_periods=EMA_MAJOR).mean().values
    
    # Calculate RSI
    delta = close_series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    avg_gains = gains.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_losses = losses.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    
    # Calculate Bollinger Bands
    bb_middle = close_series.rolling(window=BB_PERIOD, min_periods=BB_PERIOD).mean()
    bb_std = close_series.rolling(window=BB_PERIOD, min_periods=BB_PERIOD).std()
    bb_upper = bb_middle + BB_STD_DEV * bb_std
    bb_lower = bb_middle - BB_STD_DEV * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width = bb_width.fillna(0.0).values
    
    # Calculate ATR
    tr1 = high_series - low_series
    tr2 = (high_series - close_series.shift(1)).abs()
    tr3 = (low_series - close_series.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
    atr = atr.fillna(0.0).values
    
    # Calculate volume ratio
    volume_avg = volume_series.rolling(window=VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).mean()
    volume_ratio = volume / np.where(volume_avg > 0, volume_avg, 1.0)
    volume_ratio = np.nan_to_num(volume_ratio, nan=1.0)
    
    # Calculate ATR percentage
    atr_pct = atr / np.where(close > 0, close, 1.0)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW + 5,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        BB_PERIOD
    )
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Volatility filter (avoid extreme ATR)
        if atr_pct[i] < ATR_MIN_PCT or atr_pct[i] > ATR_MAX_PCT:
            signals[i] = 0.0
            continue
        
        # Major trend filter
        price_vs_major = (close[i] - ema_major[i]) / close[i]
        major_bullish = price_vs_major > 0.001
        major_bearish = price_vs_major < -0.001
        
        # EMA stack alignment
        ema_bullish = (
            ema_fast[i] > ema_medium[i] and
            ema_medium[i] > ema_slow[i] and
            ema_slow[i] > ema_major[i]
        )
        ema_bearish = (
            ema_fast[i] < ema_medium[i] and
            ema_medium[i] < ema_slow[i] and
            ema_slow[i] < ema_major[i]
        )
        
        # Trend strength
        trend_strength = (
            abs(ema_fast[i] - ema_medium[i]) / close[i] +
            abs(ema_medium[i] - ema_slow[i]) / close[i]
        ) / 2.0
        
        if trend_strength < 0.001:
            signals[i] = 0.0
            continue
        
        # Bollinger Band filter
        bb_ok = BB_WIDTH_LOW < bb_width[i] < BB_WIDTH_HIGH
        
        # Volume filter
        volume_ok = volume_ratio[i] >= 0.8
        
        # Calculate signal
        raw_signal = 0.0
        
        # LONG signal
        if major_bullish and ema_bullish and bb_ok and volume_ok:
            # RSI filter for long
            if RSI_LONG_MIN <= rsi[i] <= RSI_LONG_MAX:
                base_confidence = 0.5
                
                # Trend strength factor
                trend_factor = min(trend_strength / 0.005, 1.0)
                base_confidence += trend_factor * 0.3
                
                # RSI quality
                if 50 <= rsi[i] <= 65:
                    base_confidence *= 1.1
                
                # Volume boost
                if volume_ratio[i] >= VOLUME_THRESHOLD:
                    base_confidence *= 1.15
                
                raw_signal = base_confidence
        
        # SHORT signal
        elif major_bearish and ema_bearish and bb_ok and volume_ok:
            # RSI filter for short
            if RSI_SHORT_MIN <= rsi[i] <= RSI_SHORT_MAX:
                base_confidence = 0.5
                
                trend_factor = min(trend_strength / 0.005, 1.0)
                base_confidence += trend_factor * 0.3
                
                # RSI quality
                if 35 <= rsi[i] <= 50:
                    base_confidence *= 1.1
                
                # Volume boost
                if volume_ratio[i] >= VOLUME_THRESHOLD:
                    base_confidence *= 1.15
                
                raw_signal = -base_confidence
        
        # Apply volatility adjustment
        if raw_signal != 0.0:
            vol_adjust = min(1.5, 0.015 / max(atr_pct[i], 0.001))
            signal = raw_signal * vol_adjust
            
            # Apply thresholds
            if abs(signal) >= MIN_SIGNAL:
                signals[i] = np.clip(signal, -MAX_SIGNAL, MAX_SIGNAL)
    
    # Smooth signals
    if n >= SIGNAL_SMOOTHING:
        signal_series = pd.Series(signals)
        signals = signal_series.ewm(
            span=SIGNAL_SMOOTHING, 
            adjust=False, 
            min_periods=SIGNAL_SMOOTHING
        ).mean().values
        signals = np.nan_to_num(signals, nan=0.0)
    
    # Final clip
    signals = np.clip(signals, -1.0, 1.0)
    
    return signals