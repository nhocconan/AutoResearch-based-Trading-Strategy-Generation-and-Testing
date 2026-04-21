#!/usr/bin/env python3
"""
6h_1d_Camarilla_R3S3_Fade_R4S4_Breakout_Volume_Filter
Hypothesis: Daily Camarilla pivot levels R3/S3 act as mean-reversion zones, while R4/S4 indicate breakout strength. Fade at R3/S3 with volume divergence, breakout at R4/S4 with volume confirmation. Designed for low trade frequency (target: 12-37/year) to minimize fee drag in 6h timeframe. Works in both bull and bear markets by adapting to regime via price action at key levels.
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
    # R3 = P + (Range * 1.1000)
    # S3 = P - (Range * 1.1000)
    # R4 = P + (Range * 1.5000)
    # S4 = P - (Range * 1.5000)
    P = (high_daily + low_daily + close_daily) / 3.0
    range_daily = high_daily - low_daily
    r3_daily = P + (range_daily * 1.1000)
    s3_daily = P - (range_daily * 1.1000)
    r4_daily = P + (range_daily * 1.5000)
    s4_daily = P - (range_daily * 1.5000)
    
    # Align daily Camarilla levels to 6h timeframe
    r3_daily_aligned = align_htf_to_ltf(prices, df_daily, r3_daily)
    s3_daily_aligned = align_htf_to_ltf(prices, df_daily, s3_daily)
    r4_daily_aligned = align_htf_to_ltf(prices, df_daily, r4_daily)
    s4_daily_aligned = align_htf_to_ltf(prices, df_daily, s4_daily)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 24-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 24:
            volume_avg[i] = np.mean(volume[i-24:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if NaN in critical values
        if (np.isnan(r3_daily_aligned[i]) or np.isnan(s3_daily_aligned[i]) or 
            np.isnan(r4_daily_aligned[i]) or np.isnan(s4_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3 = r3_daily_aligned[i]
        s3 = s3_daily_aligned[i]
        r4 = r4_daily_aligned[i]
        s4 = s4_daily_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Fade at R3/S3: mean reversion from extreme levels
            # Long: price rejects S3 with volume confirmation (buying pressure)
            if price > s3 and price < (s3 + (r3 - s3) * 0.2) and vol_ok:
                # Additional confirmation: price closing near high of bar
                if close[i] > (high[i] + low[i]) / 2:
                    signals[i] = 0.25
                    position = 1
            # Short: price rejects R3 with volume confirmation (selling pressure)
            elif price < r3 and price > (r3 - (r3 - s3) * 0.2) and vol_ok:
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
            # Long exit: price returns to P (mean reversion) or breaks S4 (failed breakout)
            P_daily = (high_daily + low_daily + close_daily) / 3.0
            P_aligned = align_htf_to_ltf(prices, df_daily, P_daily)
            if not np.isnan(P_aligned[i]):
                if price < P_aligned[i] or price < s4:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to P (mean reversion) or breaks R4 (failed breakdown)
            P_daily = (high_daily + low_daily + close_daily) / 3.0
            P_aligned = align_htf_to_ltf(prices, df_daily, P_daily)
            if not np.isnan(P_aligned[i]):
                if price > P_aligned[i] or price > r4:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Camarilla_R3S3_Fade_R4S4_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0