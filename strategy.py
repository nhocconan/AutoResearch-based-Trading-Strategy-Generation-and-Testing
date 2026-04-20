#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points: PP = (H+L+C)/3, S1 = 2*PP - H, R1 = 2*PP - L
    pp_1w = (high_1w + low_1w + close_1w) / 3
    s1_1w = 2 * pp_1w - high_1w
    r1_1w = 2 * pp_1w - low_1w
    
    # Align weekly pivots to 6h
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    
    # Load daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Daily volume ratio (current / 20-period average)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma_20_1d == 0, 1, vol_ma_20_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 6h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        pp = pp_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        
        # Volume filter: require above-average volume
        vol_filter = vol_ratio > 1.2
        
        if position == 0:
            # Enter long when price crosses above weekly R1 with volume
            if price > r1 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short when price crosses below weekly S1 with volume
            elif price < s1 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly PP or volatility spike
            if price < pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly PP or volatility spike
            if price > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1S1_Breakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0