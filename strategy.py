#!/usr/bin/env python3
"""
12h_1d_Pivot_R2S2_Breakout_Volume
Hypothesis: Daily pivot points R2/S2 provide strong support/resistance. Breakouts with volume confirmation capture significant directional moves while minimizing trades. Designed for 12h timeframe to reduce frequency and avoid fee drag. Works in bull via upside breakouts and bear via downside breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for pivot points
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily pivot points: P = (H+L+C)/3, R2 = P + (H-L), S2 = P - (H-L)
    pivot_daily = (high_daily + low_daily + close_daily) / 3.0
    r2_daily = pivot_daily + (high_daily - low_daily)
    s2_daily = pivot_daily - (high_daily - low_daily)
    
    # Align daily pivot levels to 12h timeframe
    r2_daily_aligned = align_htf_to_ltf(prices, df_daily, r2_daily)
    s2_daily_aligned = align_htf_to_ltf(prices, df_daily, s2_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if NaN in critical values
        if (np.isnan(r2_daily_aligned[i]) or np.isnan(s2_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r2 = r2_daily_aligned[i]
        s2 = s2_daily_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above R2 with volume confirmation
            if price > r2 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S2 with volume confirmation
            elif price < s2 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S2 (failed breakout)
            if price < s2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R2 (failed breakdown)
            if price > r2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Pivot_R2S2_Breakout_Volume"
timeframe = "12h"
leverage = 1.0