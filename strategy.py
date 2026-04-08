#!/usr/bin/env python3
# [24995] 6h_1w_pivot_breakout_v1
# Hypothesis: 6-hour breakouts from weekly Pivot Point (PP) levels with daily volume confirmation.
# Long when price breaks above weekly R1 with volume > 1.3x daily average and close > weekly PP.
# Short when price breaks below weekly S1 with volume > 1.3x daily average and close < weekly PP.
# Exit when price returns to weekly PP. Uses weekly pivot for trend bias to avoid counter-trend trades.
# Designed to generate ~15-35 trades/year to avoid fee flood while capturing institutional levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Pivot Point calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Pivot Point levels: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pp = np.full(len(df_1w), np.nan)
    r1 = np.full(len(df_1w), np.nan)
    s1 = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        pp[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        r1[i] = 2 * pp[i] - low_1w[i]
        s1[i] = 2 * pp[i] - high_1w[i]
    
    # Get daily data for volume moving average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day volume moving average
    volume_1d = df_1d['volume'].values
    vol_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-20:i])
    
    # Align weekly PP, R1, S1 to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Align daily 20-period volume MA to 6h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to weekly PP
            if price <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to weekly PP
            if price >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly R1 with volume expansion and above weekly PP
            if price > r1_aligned[i] and vol_ratio > 1.3 and price > pp_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly S1 with volume expansion and below weekly PP
            elif price < s1_aligned[i] and vol_ratio > 1.3 and price < pp_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals