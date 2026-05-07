#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot calculation and trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    donchian_high = pd.Series(high_w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_w, donchian_low)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_w_shifted = np.roll(high_w, 1)
    low_w_shifted = np.roll(low_w, 1)
    close_w_shifted = np.roll(df_w['close'].values, 1)
    
    pivot = (high_w_shifted + low_w_shifted + close_w_shifted) / 3
    r1 = 2 * pivot - low_w_shifted
    s1 = 2 * pivot - high_w_shifted
    r2 = pivot + (high_w_shifted - low_w_shifted)
    s2 = pivot - (high_w_shifted - low_w_shifted)
    r3 = high_w_shifted + 2 * (pivot - low_w_shifted)
    s3 = low_w_shifted - 2 * (high_w_shifted - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_w, s3)
    
    # Calculate volume confirmation (current volume vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high AND above weekly R3 pivot, with volume confirmation
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > r3_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low AND below weekly S3 pivot, with volume confirmation
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < s3_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low (trend reversal)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high (trend reversal)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals