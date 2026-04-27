#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend confirmation and volume spike.
- Uses Camarilla pivot levels from daily high/low/close (R3/S3 for breakout, R4/S4 for continuation)
- Requires 1d EMA34 trend filter to avoid counter-trend trades
- Volume confirmation: current volume > 1.8x 20-period average to ensure institutional participation
- Designed for 6h timeframe to capture multi-day moves while avoiding excessive noise
- Target: 15-30 trades/year (60-120 over 4 years) to minimize fee drag
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
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = C + (H-L) * 1.1/2
    # S3 = C - (H-L) * 1.1/2
    # R4 = C + (H-L) * 1.1
    # S4 = C - (H-L) * 1.1
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.1 / 2
    s3_1d = close_1d - range_1d * 1.1 / 2
    r4_1d = close_1d + range_1d * 1.1
    s4_1d = close_1d - range_1d * 1.1
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.8 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for EMA
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema34_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and above 1d EMA34
            if (close[i] > r3_6h[i] and volume_spike[i] and 
                close[i] > ema34_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume spike and below 1d EMA34
            elif (close[i] < s3_6h[i] and volume_spike[i] and 
                  close[i] < ema34_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below S3 OR below 1d EMA34 (trend change)
            if (close[i] < s3_6h[i] or close[i] < ema34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 OR above 1d EMA34 (trend change)
            if (close[i] > r3_6h[i] or close[i] > ema34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0