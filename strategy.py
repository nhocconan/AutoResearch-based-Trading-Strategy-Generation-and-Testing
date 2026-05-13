#!/usr/bin/env python3
"""
4h_Camilla_Pivot_Breakout_12hTrend_VolumeFilter
Hypothesis: Use daily Camarilla pivot levels (S1/S3 for longs, R1/R3 for shorts) as breakout levels.
Enter long when price breaks above S3 with volume > 1.5x 20-period average and 12h EMA50 trending up (close > EMA50).
Enter short when price breaks below R1 with volume > 1.5x 20-period average and 12h EMA50 trending down (close < EMA50).
Exit when price returns to the pivot level (PP) or reverses direction.
Designed for 4h timeframe to capture multi-day swings with limited trades (target 20-50/year).
Works in bull markets by catching breakouts and in bear markets by fading false breaks at resistance.
"""

name = "4h_Camilla_Pivot_Breakout_12hTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1 / 12
    # S2 = C - (H - L) * 1.1 / 6
    # S3 = C - (H - L) * 1.1 / 4
    # R1 = C + (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    # R3 = C + (H - L) * 1.1 / 4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    s1 = close_1d - range_1d * 1.1 / 12
    s2 = close_1d - range_1d * 1.1 / 6
    s3 = close_1d - range_1d * 1.1 / 4
    r1 = close_1d + range_1d * 1.1 / 12
    r2 = close_1d + range_1d * 1.1 / 6
    r3 = close_1d + range_1d * 1.1 / 4
    
    # Align pivot levels to 4h timeframe (use previous day's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above S3 with volume spike and 12h EMA50 trending up
            if close[i] > s3_aligned[i] and vol_spike and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below R1 with volume spike and 12h EMA50 trending down
            elif close[i] < r1_aligned[i] and vol_spike and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot level or breaks below S1
            if close[i] < pp_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot level or breaks above R1
            if close[i] > pp_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals