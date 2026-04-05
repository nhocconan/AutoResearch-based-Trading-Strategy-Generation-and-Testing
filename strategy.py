#!/usr/bin/env python3
"""
Experiment #9931: 6h 12-Hour Range Breakout with Volume Confirmation
Hypothesis: Price breaks above/below the prior 12-hour high/low (12h range) with volume confirmation
captures momentum bursts. Works in bull markets (breakouts above recent highs) and bear markets 
(breakdowns below recent lows). Volume filter reduces false breakouts. 
Target: 80-160 total trades over 4 years (20-40/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9931_6h_12h_range_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
RANGE_PERIOD = 2  # 2 periods of 6h = 12h
VOLUME_SPIKE_MULTIPLIER = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

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
    
    # 12-hour range (2 periods back)
    range_high = pd.Series(high).rolling(window=RANGE_PERIOD, min_periods=RANGE_PERIOD).max().shift(1).values
    range_low = pd.Series(low).rolling(window=RANGE_PERIOD, min_periods=RANGE_PERIOD).min().shift(1).values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RANGE_PERIOD + 1, 20) + 1
    
    for i in range(start, n):
        # Skip if range data not available
        if np.isnan(range_high[i]) or np.isnan(range_low[i]):
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
        
        # Breakout conditions: above prior 12h high or below prior 12h low
        bullish_breakout = close[i] > range_high[i] if not np.isnan(range_high[i]) else False
        bearish_breakout = close[i] < range_low[i] if not np.isnan(range_low[i]) else False
        
        # Generate signals
        if position == 0:
            if bullish_breakout and volume_spike:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bearish_breakout and volume_spike:
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