#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d volume confirmation and 1w trend filter
# Designed for low trade frequency (target 15-25/year) with mean-reversion logic
# Works in both bull (fade overextended moves) and bear (fade dead cat bounces) markets
# Uses Camarilla levels (H3/L3) from 1d, volume spike to confirm interest, and 1w EMA for trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels (H3, L3)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H3 = H + 1.1 * Range / 2
    # L3 = L - 1.1 * Range / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h3_1d = high_1d + 1.1 * range_1d / 2.0
    l3_1d = low_1d - 1.1 * range_1d / 2.0
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: price touches L3 (support) + uptrend + volume spike
        if (low[i] <= l3_1d_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume[i] > 1.5 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price touches H3 (resistance) + downtrend + volume spike
        elif (high[i] >= h3_1d_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume[i] > 1.5 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price moves back to pivot
        elif position == 1 and (close[i] >= pivot_1d_aligned[i] if 'pivot_1d_aligned' in locals() else close[i] >= ((high_1d + low_1d + close_1d) / 3.0)):
            # Simplified: exit when price returns to average of H3/L3
            mid_point = (h3_1d_aligned[i] + l3_1d_aligned[i]) / 2.0
            if close[i] >= mid_point:
                position = 0
                signals[i] = 0.0
        elif position == -1 and close[i] <= (h3_1d_aligned[i] + l3_1d_aligned[i]) / 2.0:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_1dVolume_1wEMA_Reversal"
timeframe = "12h"
leverage = 1.0