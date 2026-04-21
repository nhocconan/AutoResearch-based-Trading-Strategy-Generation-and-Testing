#!/usr/bin/env python3
"""
4h_1D_Camarilla_R1S1_Breakout_Volume_Filter
Hypothesis: Daily Camarilla pivot levels R1/S1 act as mean-reversion zones, while R4/S4 indicate breakout strength. Fade at R1/S1 with volume confirmation, breakout at R4/S4 with volume confirmation. Designed for low trade frequency (target: 12-37/year) to minimize fee drag in 4h timeframe. Works in both bull and bear markets by adapting to regime via price action at key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Camarilla pivot points
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily Camarilla pivot levels
    # P = (H + L + C) / 3
    # Range = H - L
    # R1 = P + (Range * 0.382)
    # S1 = P - (Range * 0.382)
    # R4 = P + (Range * 1.5000)
    # S4 = P - (Range * 1.5000)
    P = (high_daily + low_daily + close_daily) / 3.0
    range_daily = high_daily - low_daily
    r1_daily = P + (range_daily * 0.382)
    s1_daily = P - (range_daily * 0.382)
    r4_daily = P + (range_daily * 1.5000)
    s4_daily = P - (range_daily * 1.5000)
    
    # Align daily Camarilla levels to 4h timeframe
    r1_daily_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    r4_daily_aligned = align_htf_to_ltf(prices, df_daily, r4_daily)
    s4_daily_aligned = align_htf_to_ltf(prices, df_daily, s4_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 6-period average (6*4h = 1 day)
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 6:
            volume_avg[i] = np.mean(volume[i-6:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(6, n):
        # Skip if NaN in critical values
        if (np.isnan(r1_daily_aligned[i]) or np.isnan(s1_daily_aligned[i]) or 
            np.isnan(r4_daily_aligned[i]) or np.isnan(s4_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = r1_daily_aligned[i]
        s1 = s1_daily_aligned[i]
        r4 = r4_daily_aligned[i]
        s4 = s4_daily_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Fade at R1/S1: mean reversion from extreme levels
            # Long: price rejects S1 with volume confirmation (buying pressure)
            if price > s1 and price < (s1 + (r1 - s1) * 0.3) and vol_ok:
                # Additional confirmation: price closing near high of bar
                if close[i] > (high[i] + low[i]) / 2:
                    signals[i] = 0.25
                    position = 1
            # Short: price rejects R1 with volume confirmation (selling pressure)
            elif price < r1 and price > (r1 - (r1 - s1) * 0.3) and vol_ok:
                # Additional confirmation: price closing near low of bar
                if close[i] < (high[i] + low[i]) / 2:
                    signals[i] = -0.25
                    position = -1
            # Breakout at R4/S4: strong momentum continuation
            # Long: price breaks above R4 with volume
            elif price > r4 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume
            elif price < s4 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to S1 (mean reversion) or breaks S4 (failed breakout)
            if price < s1 or price > r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to R1 (mean reversion) or breaks S4 (failed breakdown)
            if price > r1 or price < s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1D_Camarilla_R1S1_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0