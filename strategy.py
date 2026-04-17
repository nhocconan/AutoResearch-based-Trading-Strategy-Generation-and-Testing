#!/usr/bin/env python3
"""
12h_382_Retracement_Trend_Entry
Hypothesis: On 12h, enter long at 38.2% Fibonacci retracement of prior swing low to high when price is above weekly EMA20 (uptrend), enter short at 61.8% retracement of prior swing high to low when price is below weekly EMA20 (downtrend). Uses weekly trend filter to avoid counter-trend trades. Fibonacci levels act as institutional support/resistance in trending markets. Designed for 15-25 trades/year to minimize fee drag and work in both bull/bear regimes via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Weekly data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === Daily data for swing points ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate swing high and low using 5-day window
    # Swing high: highest high in last 5 days
    # Swing low: lowest low in last 5 days
    window = 5
    swing_high = pd.Series(high_1d).rolling(window=window, min_periods=window).max().values
    swing_low = pd.Series(low_1d).rolling(window=window, min_periods=window).min().values
    
    # Calculate Fibonacci retracement levels
    # 38.2% level for longs (retracement from swing low to swing high)
    # 61.8% level for shorts (retracement from swing high to swing low)
    fib_range = swing_high - swing_low
    fib_382 = swing_low + 0.382 * fib_range  # Long entry level
    fib_618 = swing_high - 0.382 * fib_range  # Short entry level (same as swing_low + 0.618*fib_range)
    
    # Align to 12h timeframe
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, swing_low)
    fib_382_aligned = align_htf_to_ltf(prices, df_1d, fib_382)
    fib_618_aligned = align_htf_to_ltf(prices, df_1d, fib_618)
    
    signals = np.zeros(n)
    
    # Warmup: covers weekly EMA20 and daily swing calculations
    warmup = max(20, 5)  # 20 for weekly EMA, 5 for daily swing
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(fib_382_aligned[i]) or 
            np.isnan(fib_618_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: price at 38.2% retracement + above weekly EMA20 (uptrend)
            if abs(close[i] - fib_382_aligned[i]) < 0.001 * close[i] and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price at 61.8% retracement + below weekly EMA20 (downtrend)
            elif abs(close[i] - fib_618_aligned[i]) < 0.001 * close[i] and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse when price reaches opposite Fibonacci level
        elif position == 1:
            if abs(close[i] - fib_618_aligned[i]) < 0.001 * close[i]:  # exit long at 61.8%
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if abs(close[i] - fib_382_aligned[i]) < 0.001 * close[i]:  # exit short at 38.2%
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_382_Retracement_Trend_Entry"
timeframe = "12h"
leverage = 1.0