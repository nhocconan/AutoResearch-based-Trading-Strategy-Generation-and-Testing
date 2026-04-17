#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Confirm
Hypothesis: On 12h timeframe, use daily Camarilla pivot levels (R1/S1) with volume confirmation and trend filter. 
Enter long when price breaks above R1 with above-average volume; short when breaks below S1 with volume. 
Use weekly trend filter (price above/below weekly SMA50) to avoid counter-trend trades. 
Designed for low trade frequency (<30/year) to minimize fee drag and work in both bull/bear markets via trend alignment.
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
    
    # === Daily data for Camarilla pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    range_1d = high_1d - low_1d
    camarilla_R1 = close_1d + range_1d * 1.1 / 12
    camarilla_S1 = close_1d - range_1d * 1.1 / 12
    
    # Align daily levels to 12h (will use previous day's levels)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # === Weekly trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly SMA50 for trend filter
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # === Daily volume average for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers 20-day volume average, 50-week SMA
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(sma50_1w_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.2x 20-day average
        vol_filter = vol_1d_current > 1.2 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: price above/below weekly SMA50
        above_weekly_trend = close[i] > sma50_1w_aligned[i]
        below_weekly_trend = close[i] < sma50_1w_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: price > Camarilla R1 + volume + above weekly trend
            if close[i] > camarilla_R1_aligned[i] and vol_filter and above_weekly_trend:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < Camarilla S1 + volume + below weekly trend
            elif close[i] < camarilla_S1_aligned[i] and vol_filter and below_weekly_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse signal at opposite Camarilla level
        elif position == 1:
            if close[i] < camarilla_S1_aligned[i]:  # break below S1 = exit long
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > camarilla_R1_aligned[i]:  # break above R1 = exit short
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Confirm"
timeframe = "12h"
leverage = 1.0