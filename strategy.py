#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Pivot_Breakout_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d pivot points (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_1d = prev_high - prev_low
    
    # Support and resistance levels
    r1 = pivot + (range_1d * 0.382)  # Fibonacci 0.382
    s1 = pivot - (range_1d * 0.382)
    r2 = pivot + (range_1d * 0.618)  # Fibonacci 0.618
    s2 = pivot - (range_1d * 0.618)
    
    # Align pivot levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: 4h volume > 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R2 with trend filter and volume
            if close[i] > r2_4h[i] and close[i] > ema50_1d_aligned[i] and volume[i] > vol_ma20[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S2 with trend filter and volume
            elif close[i] < s2_4h[i] and close[i] < ema50_1d_aligned[i] and volume[i] > vol_ma20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 or trend reverses
            if close[i] < s1_4h[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R1 or trend reverses
            if close[i] > r1_4h[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h strategy using 1d Fibonacci pivot levels (0.382, 0.618) for breakout entries
# in the direction of the 1d EMA50 trend, with volume confirmation. Exits at opposite
# pivot levels or trend reversal. Designed to work in both bull and bear markets by
# following the daily trend while using institutional pivot levels for entry/exit.
# Targets 30-80 trades over 4 years (7-20/year) to minimize fee drift. Uses discrete
# sizing (0.25) to reduce churn. Works on BTC/ETH via institutional pivot levels.