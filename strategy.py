#!/usr/bin/env python3
"""
Experiment #10211: 6h 123 Reversal Pattern with Volume Confirmation
Hypothesis: The 123 reversal pattern (test of prior swing high/low) combined with volume confirmation
provides high-probability reversal entries at key swing levels. Works in both bull and bear markets
by capturing mean reversion at swing extremes. Target: 80-180 total trades over 4 years (20-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10211_6h_123_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
LOOKBACK_PERIOD = 10
VOLUME_SPIKE_MULTIPLIER = 1.8
SIGNAL_SIZE = 0.28
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_swing_points(high, low, lookback):
    """Calculate swing high and low points"""
    swing_high = np.full_like(high, np.nan)
    swing_low = np.full_like(low, np.nan)
    
    for i in range(lookback, len(high)):
        # Swing high: highest high in lookback period
        swing_high[i] = np.max(high[i-lookback:i+1])
        # Swing low: lowest low in lookback period
        swing_low[i] = np.min(low[i-lookback:i+1])
    
    return swing_high, swing_low

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Swing points for 123 pattern
    swing_high, swing_low = calculate_swing_points(high, low, LOOKBACK_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = LOOKBACK_PERIOD + 1
    
    for i in range(start, n):
        # Skip if swing points not available
        if np.isnan(swing_high[i]) or np.isnan(swing_low[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # 123 Reversal pattern:
        # For bullish reversal: price tests swing low (point 2) and bounces with volume
        # For bearish reversal: price tests swing high (point 2) and rejects with volume
        near_swing_low = abs(close[i] - swing_low[i]) <= (0.5 * atr[i])  # Within 0.5 ATR of swing low
        near_swing_high = abs(close[i] - swing_high[i]) <= (0.5 * atr[i])  # Within 0.5 ATR of swing high
        
        # Bullish reversal: test of swing low with bounce
        bullish_reversal = near_swing_low and (close[i] > low[i]) and volume_spike
        # Bearish reversal: test of swing high with rejection
        bearish_reversal = near_swing_high and (close[i] < high[i]) and volume_spike
        
        # Generate signals
        if position == 0:
            if bullish_reversal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bearish_reversal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals