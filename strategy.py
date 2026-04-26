#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA50 trend filter and volume spike. Designed for low trade frequency (~15-25/year) to minimize fee drag while capturing strong trending moves. Works in bull markets via long breakouts and in bear markets via short breakdowns, with volume confirmation ensuring institutional participation. Uses discrete sizing 0.25 to balance return and risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1, S1, R2, S2 for each 1d bar
    rng = high_1d - low_1d
    r1 = close_1d + 1.125 * rng * (1/6)  # R1 = C + 1.125*(H-L)/6
    s1 = close_1d - 1.125 * rng * (1/6)  # S1 = C - 1.125*(H-L)/6
    r2 = close_1d + 1.125 * rng * (2/6)  # R2 = C + 1.125*(H-L)/3
    s2 = close_1d - 1.125 * rng * (2/6)  # S2 = C - 1.125*(H-L)/3
    
    # Align to 1d timeframe (wait for 1d bar to close)
    r1_1d = align_htf_to_ltf(prices, df_1d, r1)
    s1_1d = align_htf_to_ltf(prices, df_1d, s1)
    r2_1d = align_htf_to_ltf(prices, df_1d, r2)
    s2_1d = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume spike: volume > 2.0x 20-period median volume (stricter for lower frequency)
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 1w EMA, 20 for volume median
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(r1_1d[i]) or
            np.isnan(s1_1d[i]) or
            np.isnan(r2_1d[i]) or
            np.isnan(s2_1d[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price > R1 and volume spike, in uptrend (close > EMA50)
            long_entry = (close_val > r1_1d[i]) and vol_spike and (close_val > ema_50_val)
            # Short: price < S1 and volume spike, in downtrend (close < EMA50)
            short_entry = (close_val < s1_1d[i]) and vol_spike and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or at R2 (take profit)
            if close_val < ema_50_val or close_val > r2_1d[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or at S2 (take profit)
            if close_val > ema_50_val or close_val < s2_1d[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0