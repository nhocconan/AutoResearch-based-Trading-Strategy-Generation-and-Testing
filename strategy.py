#!/usr/bin/env python3
"""
4h_Donchian20_1dVolume_Conservative_v1
Concept: 4h Donchian(20) breakout with 1d volume confirmation and EMA200 trend filter.
- Long: Close > 20-period high AND volume > 1d average volume AND price > EMA200
- Short: Close < 20-period low AND volume > 1d average volume AND price < EMA200
- Exit: Opposite Donchian breakout or price crosses EMA200
- Position sizing: 0.25
- Target: 20-40 trades/year (80-160 total over 4 years)
- Works in bull/bear: EMA200 defines trend, Donchian captures breakouts, volume filters weak moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1dVolume_Conservative_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 4h: EMA200 trend filter ===
    close = prices['close'].values
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # === 4h: Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Daily: Average volume (20-period) ===
    vol_1d = df_1d['volume'].values
    avg_vol_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for EMA200
    
    for i in range(start_idx, n):
        # Get values
        ema200_val = ema200[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        vol_avg = avg_vol_20_aligned[i]
        vol_cur = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema200_val) or np.isnan(dh) or np.isnan(dl) or 
            np.isnan(vol_avg)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + volume confirmation + above EMA200
            if close[i] > dh and vol_cur > vol_avg and close[i] > ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + volume confirmation + below EMA200
            elif close[i] < dl and vol_cur > vol_avg and close[i] < ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Break below Donchian low OR price crosses below EMA200
            if close[i] < dl or close[i] < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Break above Donchian high OR price crosses above EMA200
            if close[i] > dh or close[i] > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals