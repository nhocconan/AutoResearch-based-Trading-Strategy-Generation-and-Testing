#!/usr/bin/env python3
"""
12h_1W_1D_Camarilla_R1_S1_Breakout_With_Volume_Spike
Hypothesis: On 12h timeframe, breakouts above Camarilla R1 level (from prior 1d) or below S1 level (from prior 1d) with volume spike (>2x median) and trend filter (1w EMA50) provide high-probability entries. Uses 12h close for signal generation. Designed for low trade frequency (12-37/year) to minimize fee drag while capturing breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d and 1w HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Camarilla pivot levels (based on prior day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align 1d levels to 12h timeframe (use prior day's levels for today's trading)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1w EMA50 trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Volume spike detection (12h) ===
    volume = prices['volume'].values
    # Calculate median volume over last 28 periods (14 days)
    vol_median = np.full_like(volume, np.nan)
    for i in range(28, len(volume)):
        vol_median[i] = np.median(volume[i-28:i])
    volume_spike = volume > (vol_median * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike and above 1w EMA50 (uptrend)
            if (price_close > r1_val and 
                vol_spike and 
                price_close > ema_50_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume spike and below 1w EMA50 (downtrend)
            elif (price_close < s1_val and 
                  vol_spike and 
                  price_close < ema_50_val):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses back through the pivot level
            pivot_val = (high_1d[i//16] + low_1d[i//16] + close_1d[i//16]) / 3.0 if i >= 16 else np.nan
            if position == 1 and price_close < pivot_val:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_1W_1D_Camarilla_R1_S1_Breakout_With_Volume_Spike"
timeframe = "12h"
leverage = 1.0