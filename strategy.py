#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_v1
Breakout of weekly pivot R3/S3 levels with volume confirmation.
Long: close > weekly R3 + volume > 1.5x avg volume + price > 200EMA
Short: close < weekly S3 + volume > 1.5x avg volume + price < 200EMA
Exit when price returns to weekly pivot (PP) or opposite pivot level.
Designed to capture strong moves after weekly pivot breakouts.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Pivot Points (using weekly OHLC) ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate pivot points from weekly OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # === 200 EMA for trend filter ===
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # === Volume filter: 1.5x average volume (20-period) ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * avg_volume
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_200[i]) or 
            np.isnan(volume_threshold[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: breakout above R3 with volume confirmation and above 200EMA
            if (close[i] > r3_aligned[i] and 
                volume[i] > volume_threshold[i] and 
                close[i] > ema_200[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: breakdown below S3 with volume confirmation and below 200EMA
            elif (close[i] < s3_aligned[i] and 
                  volume[i] > volume_threshold[i] and 
                  close[i] < ema_200[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to pivot point (PP) or breaks below S3
            if (close[i] <= pp_aligned[i] or 
                close[i] < s3_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point (PP) or breaks above R3
            if (close[i] >= pp_aligned[i] or 
                close[i] > r3_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Breakout_v1"
timeframe = "6h"
leverage = 1.0