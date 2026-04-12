#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Volume_Regime_v1
Hypothesis: On 12h timeframe, enter long when price breaks above daily Camarilla R3 with volume confirmation and chop regime favors trending, enter short when price breaks below daily Camarilla S3 with volume confirmation and chop regime favors trending. Uses daily Camarilla levels for structure, volume filter for institutional participation, and chop regime to avoid whipsaws in sideways markets. Target: 15-30 trades per year per symbol (60-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY INDICATORS: Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla R3 and S3 levels (key reversal levels)
    r3 = close_1d + range_1d * 1.1 / 4
    s3 = close_1d - range_1d * 1.1 / 4
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === VOLUME FILTER ===
    vol_ma = np.zeros_like(volume)
    if len(volume) >= 20:
        vol_ma[20] = np.mean(volume[0:20])
        for i in range(21, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    volume_filter = volume > 1.5 * vol_ma
    
    # === CHOPPINESS INDEX REGIME FILTER (using daily data) ===
    # Chop = 100 * log10(sum(ATR(14)) / (highest_high - lowest_low)) / log10(14)
    tr = np.maximum(high_1d - low_1d, np.maximum(abs(high_1d - np.roll(close_1d, 1)), abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # first period
    atr14 = np.zeros_like(tr)
    if len(tr) >= 14:
        atr14[14] = np.mean(tr[0:14])
        for i in range(15, len(tr)):
            atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    
    # Calculate choppy index
    sum_atr14 = np.zeros_like(atr14)
    if len(atr14) >= 14:
        sum_atr14[14] = np.sum(atr14[0:14])
        for i in range(15, len(sum_atr14)):
            sum_atr14[i] = sum_atr14[i-1] - atr14[i-14] + atr14[i]
    
    highest_high = np.maximum.accumulate(high_1d)
    lowest_low = np.minimum.accumulate(low_1d)
    hh_ll_diff = highest_high - lowest_low
    
    chop = np.zeros_like(close_1d)
    for i in range(14, len(chop)):
        if sum_atr14[i] > 0 and hh_ll_diff[i] > 0:
            chop[i] = 100 * np.log10(sum_atr14[i] / hh_ll_diff[i]) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Align chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Chop regime: < 38.2 = trending, > 61.8 = ranging
    chop_trending = chop_aligned < 38.2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume and chop regime filter
        long_breakout = (close[i] > r3_aligned[i]) and volume_filter[i] and chop_trending[i]
        short_breakout = (close[i] < s3_aligned[i]) and volume_filter[i] and chop_trending[i]
        
        # Exit conditions: reversal back inside Camarilla H3-L3 range or chop becomes ranging
        h3 = close_1d + range_1d * 1.1 / 2
        l3 = close_1d - range_1d * 1.1 / 2
        h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
        l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
        
        exit_long = (close[i] < h3_aligned[i]) or (close[i] > l3_aligned[i]) or (~chop_trending[i])
        exit_short = (close[i] > l3_aligned[i]) or (close[i] < h3_aligned[i]) or (~chop_trending[i])
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals