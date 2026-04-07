#!/usr/bin/env python3
"""
6H Fibonacci Retracement Pullback with 12h Trend Filter and Volume Confirmation
Long when price pulls back to 61.8% Fib level in 12h uptrend with volume confirmation
Short when price pulls back to 38.2% Fib level in 12h downtrend with volume confirmation
Exit when price crosses 50% Fib level
Uses Fibonacci retracements from recent swing highs/lows to catch trend continuations.
Works in both bull and bear markets by following the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_fib_pullback_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Swing detection (5-period lookback) ===
    # Find recent swing high and low
    swing_high = np.zeros(n)
    swing_low = np.zeros(n)
    
    for i in range(5, n):
        # Swing high: highest high in last 5 bars
        swing_high[i] = np.max(high[i-5:i+1])
        # Swing low: lowest low in last 5 bars
        swing_low[i] = np.min(low[i-5:i+1])
    
    # === Fibonacci levels ===
    diff = swing_high - swing_low
    fib_382 = swing_low + 0.382 * diff  # 38.2% retracement
    fib_500 = swing_low + 0.500 * diff  # 50% retracement
    fib_618 = swing_low + 0.618 * diff  # 61.8% retracement
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 12h trend filter (EMA 34) ===
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(swing_high[i]) or np.isnan(swing_low[i]) or 
            np.isnan(fib_382[i]) or np.isnan(fib_500[i]) or np.isnan(fib_618[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 50% Fib level
            if close[i] < fib_500[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 50% Fib level
            if close[i] > fib_500[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (above average)
            if vol_ratio[i] < 1.1:
                signals[i] = 0.0
                continue
            
            # Entry: Fibonacci pullback with 12h trend filter
            # Long: pullback to 61.8% in uptrend
            if (close[i] <= fib_618[i] * 1.001 and close[i] >= fib_618[i] * 0.999 and
                ema_12h_aligned[i] > ema_12h_aligned[i-1]):
                # Near 61.8% Fib level with rising 12h EMA -> long
                position = 1
                signals[i] = 0.25
            # Short: pullback to 38.2% in downtrend
            elif (close[i] <= fib_382[i] * 1.001 and close[i] >= fib_382[i] * 0.999 and
                  ema_12h_aligned[i] < ema_12h_aligned[i-1]):
                # Near 38.2% Fib level with falling 12h EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals