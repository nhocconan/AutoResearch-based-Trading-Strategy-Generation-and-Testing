#!/usr/bin/env python3
"""
6h Elliott Wave Fibonacci Retracement with 1d Trend Filter
Hypothesis: In trending markets (1d EMA50), price retraces to Fibonacci levels (38.2%, 50%, 61.8%) of recent swings before continuing.
Uses 6h swing high/low over 24 periods (4 days) to identify legs, then enters at retracement levels in trend direction.
Works in bull (buy 38.2/50/61.8% retracements of pullbacks in uptrend) and bear (sell at same levels in downtrend).
Target: 50-150 trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14271_6h_elliott_fibo_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate swing points (24-period = 4 days on 6h chart)
    window = 24
    highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, 21)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Fibonacci levels
    fib_levels = [0.382, 0.5, 0.618]
    
    # Start from warmup period
    start = max(window, 50, 21) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
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
        
        # Determine trend from 1d EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Calculate swing range for retracement levels
        swing_high = highest_high[i-1]  # Use previous bar's swing high
        swing_low = lowest_low[i-1]     # Use previous bar's swing low
        swing_range = swing_high - swing_low
        
        # Avoid division by zero or too small ranges
        if swing_range <= 0 or np.isnan(swing_range):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Calculate Fibonacci retracement levels
        fib_382 = swing_low + swing_range * 0.382
        fib_50 = swing_low + swing_range * 0.5
        fib_618 = swing_low + swing_range * 0.618
        
        # Entry conditions: price near Fibonacci levels in trend direction
        # Long: price at/above 38.2% and below 50% in uptrend (buy the dip)
        # Short: price at/below 61.8% and above 50% in downtrend (sell the bounce)
        tolerance = 0.001 * swing_range  # 0.1% of swing range tolerance
        
        long_entry = (uptrend and 
                     (abs(close[i] - fib_382) <= tolerance or 
                      (close[i] > fib_382 and close[i] < fib_50)))
        
        short_entry = (downtrend and 
                      (abs(close[i] - fib_618) <= tolerance or 
                       (close[i] < fib_618 and close[i] > fib_50)))
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.5 * atr[i])
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.5 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or if price breaks above 61.8% (take profit)
            if close[i] <= stop_price or close[i] >= fib_618:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or if price breaks below 38.2% (take profit)
            if close[i] >= stop_price or close[i] <= fib_382:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals