#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_WeeklyPivot_Direction
Hypothesis: On 6h timeframe, price breaking Camarilla R1/S1 levels in the direction of 1w Camarilla pivot (bullish if close > weekly pivot, bearish if close < weekly pivot) with volume confirmation (>1.8x 24-period MA) captures high-probability trend continuation moves. Weekly pivot acts as a robust trend filter resistant to whipsaw, while 6h Camarilla provides precise entries. Volume spike confirms institutional participation. Designed for 12-37 trades/year with discrete sizing (±0.25) to minimize fee drag and work in both bull/bear markets with BTC/ETH edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 24 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla R1/S1 from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    daily_range = high_1d - low_1d
    camarilla_r1 = close_1d + daily_range * 1.1 / 12
    camarilla_s1 = close_1d - daily_range * 1.1 / 12
    
    # Calculate 1w Camarilla pivot (PP) from previous 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align all HTF levels to 6h timeframe (wait for completed bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume spike filter: volume > 1.8 * 24-period MA on 6h (4 bars per day * 6 = 24)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of 1d calculation (0), 1w calculation (0), volume MA (24) + time for alignment
    # 1d -> 6h: 4 bars per day, 1w -> 6h: 28 bars per week
    start_idx = max(24, 1) + 28  # +28 to ensure 1w bar completion
    
    for i in range(start_idx, n):
        close_val = close[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        pivot_val = weekly_pivot_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any data not ready (NaN from alignment)
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(pivot_val) or 
            np.isnan(volume_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: bullish when price > weekly pivot, bearish when price < weekly pivot
        trend_bullish = close_val > pivot_val
        trend_bearish = close_val < pivot_val
        
        # 6h Camarilla breakout conditions: price breaks R1/S1 with trend alignment + volume spike
        long_breakout = close_val > r1_val
        short_breakout = close_val < s1_val
        
        long_entry = trend_bullish and long_breakout and vol_spike
        short_entry = trend_bearish and short_breakout and vol_spike
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and close_val < s1_val:  # Exit long if price breaks S1 (reversal)
            signals[i] = 0.0
            position = 0
        elif position == -1 and close_val > r1_val:  # Exit short if price breaks R1 (reversal)
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_WeeklyPivot_Direction"
timeframe = "6h"
leverage = 1.0