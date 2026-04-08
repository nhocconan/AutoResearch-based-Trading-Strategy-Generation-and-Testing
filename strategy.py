#!/usr/bin/env python3
# 6h_weekly_pivot_momentum_v2
# Hypothesis: Uses weekly pivot points as support/resistance levels with momentum confirmation.
# Enters long when price breaks above weekly R1 with bullish momentum (close > open), short when breaks below weekly S1 with bearish momentum (close < open).
# Uses 1d timeframe for pivot calculation and 6h for entry timing to avoid overtrading.
# Designed for low trade frequency (~20-50/year) to minimize fee drift and capture institutional levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_momentum_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # We'll use the last 5 days to approximate weekly OHLC
    # In practice, we'd use actual weekly data, but for simplicity we use daily
    # and calculate pivots based on prior day's range (common approximation)
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Classic pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 1
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Momentum confirmation: bullish if close > open, bearish if close < open
        bullish_momentum = close[i] > open_price[i]
        bearish_momentum = close[i] < open_price[i]
        
        # Breakout conditions
        breakout_r1 = close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]
        breakdown_s1 = close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]
        
        if position == 1:  # Long position
            # Exit: break below pivot or momentum shifts
            if close[i] < pivot_aligned[i] or not bullish_momentum:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: break above pivot or momentum shifts
            if close[i] > pivot_aligned[i] or not bearish_momentum:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: break above R1 with bullish momentum
            if breakout_r1 and bullish_momentum:
                position = 1
                signals[i] = 0.25
            # Short entry: break below S1 with bearish momentum
            elif breakdown_s1 and bearish_momentum:
                position = -1
                signals[i] = -0.25
    
    return signals