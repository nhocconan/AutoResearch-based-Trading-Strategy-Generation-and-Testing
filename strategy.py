#!/usr/bin/env python3
"""
Hypothesis: 6h Fibonacci Pivot (based on 12h range) with 1d volume spike and 1d EMA34 trend filter.
- Long when price breaks above R1 with volume > 1.5x 20-period average and close > EMA34
- Short when price breaks below S1 with volume > 1.5x 20-period average and close < EMA34
- Targets 15-25 trades/year to avoid fee drain. Works in trending markets (breakouts) and avoids chop via volume/EMA filter.
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
    volume = prices['volume'].values
    
    # === 1d EMA34 for trend filter ===
    close_series = pd.Series(close)
    ema_34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 12h high/low for Fibonacci pivot calculation ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate daily range (using previous 12h bar for pivot)
    # Pivot uses [H1, L1, C1] where H1,L1,C1 are from previous 12h bar
    range_12h = high_12h - low_12h
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    
    # Fibonacci levels
    # R1 = P + 0.382 * range
    # S1 = P - 0.382 * range
    r1_12h = pivot_12h + 0.382 * range_12h
    s1_12h = pivot_12h - 0.382 * range_12h
    
    # Align 12h levels to 6t timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # === 1d volume confirmation ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for EMA and volume MA
    warmup = 40
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34[i]) or np.isnan(r1_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        
        # Volume spike: current 1d volume > 1.5x 20-period average
        df_1d_current = get_htf_data(prices, '1d')
        vol_1d_current = df_1d_current['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d_current, vol_1d_current)
        vol_spike = vol_1d_aligned[i] > vol_ma_20_aligned[i] * 1.5
        
        # Breakout conditions
        breakout_long = price > r1_12h_aligned[i]
        breakout_short = price < s1_12h_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above R1 + volume spike + close > EMA34
            if breakout_long and vol_spike and close[i] > ema_34[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below S1 + volume spike + close < EMA34
            elif breakout_short and vol_spike and close[i] < ema_34[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite breakout
        elif position == 1:
            # Exit long if price breaks below S1
            if price < s1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above R1
            if price > r1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_FibPivot_R1S1_1dVolume1.5x_EMA34Filter"
timeframe = "6h"
leverage = 1.0