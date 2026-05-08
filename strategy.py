#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_DonchianBreakout_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel (20-period on daily)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily Donchian channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = (2 * pivot_1w) - high_1w
    s1_1w = (2 * pivot_1w) - low_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivots to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or
            np.isnan(r2_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(s2_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily Donchian high + above weekly pivot + volume confirmation
            if (close[i] > donch_high_aligned[i] and 
                close[i] > pivot_1w_aligned[i] and
                vol_ratio[i] > 1.5):
                # Avoid extreme extension beyond weekly R2
                if close[i] <= r2_1w_aligned[i] * 1.02:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below daily Donchian low + below weekly pivot + volume confirmation
            elif (close[i] < donch_low_aligned[i] and 
                  close[i] < pivot_1w_aligned[i] and
                  vol_ratio[i] > 1.5):
                # Avoid extreme extension beyond weekly S2
                if close[i] >= s2_1w_aligned[i] * 0.98:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price breaks below daily Donchian low OR below weekly pivot
            if close[i] < donch_low_aligned[i] or close[i] < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above daily Donchian high OR above weekly pivot
            if close[i] > donch_high_aligned[i] or close[i] > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals