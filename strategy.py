#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-day Camarilla Pivot + Volume Spike + Choppiness Regime Filter
# Uses daily Camarilla pivot levels (S3, S2, S1, R1, R2, R3) as support/resistance.
# Long when price touches S1/S2 with volume spike in choppy market (CHOPPINESS > 61.8).
# Short when price touches R1/R2 with volume spike in choppy market.
# Works in sideways/ranging markets by fading extremes, avoids trending markets.
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 2)
    # S2 = C - (Range * 1.1)
    # S3 = C - (Range * 1.1 * 2)
    # R1 = C + (Range * 1.1 / 2)
    # R2 = C + (Range * 1.1)
    # R3 = C + (Range * 1.1 * 2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    s1_1d = close_1d - (range_1d * 1.1 / 2)
    s2_1d = close_1d - (range_1d * 1.1)
    s3_1d = close_1d - (range_1d * 1.1 * 2)
    r1_1d = close_1d + (range_1d * 1.1 / 2)
    r2_1d = close_1d + (range_1d * 1.1)
    r3_1d = close_1d + (range_1d * 1.1 * 2)
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    s1_1d_prev = np.roll(s1_1d, 1)
    s2_1d_prev = np.roll(s2_1d, 1)
    r1_1d_prev = np.roll(r1_1d, 1)
    r2_1d_prev = np.roll(r2_1d, 1)
    s1_1d_prev[0] = np.nan
    s2_1d_prev[0] = np.nan
    r1_1d_prev[0] = np.nan
    r2_1d_prev[0] = np.nan
    
    # Align Camarilla levels to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d_prev)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d_prev)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d_prev)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d_prev)
    
    # Calculate Choppiness Index on 1d (14-period)
    # CHOP = 100 * log10(sum(ATR) / (max(HH) - min(LL))) / log10(period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    
    # Align Choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike detection: current volume > 2.0 * median of last 20 periods
    def volume_spike(vol_arr, idx):
        if idx < 20:
            return False
        median_vol = np.median(vol_arr[max(0, idx-20):idx])
        return vol_arr[idx] > 2.0 * median_vol
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(chop_aligned[i])):
            continue
        
        # Long entry: price touches S1 or S2 + volume spike + choppy market (CHOPPINESS > 61.8)
        if ((close[i] <= s1_aligned[i] * 1.001 or close[i] <= s2_aligned[i] * 1.001) and
            volume_spike(volume, i) and
            chop_aligned[i] > 61.8 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price touches R1 or R2 + volume spike + choppy market (CHOPPINESS > 61.8)
        elif ((close[i] >= r1_aligned[i] * 0.999 or close[i] >= r2_aligned[i] * 0.999) and
              volume_spike(volume, i) and
              chop_aligned[i] > 61.8 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price moves back toward pivot or chop drops below 38.2 (trending market)
        elif position == 1 and (close[i] >= pivot_1d[i] * 0.999 or chop_aligned[i] < 38.2):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= pivot_1d[i] * 1.001 or chop_aligned[i] < 38.2):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_1d_Camarilla_Pivot_Volume_Chop"
timeframe = "12h"
leverage = 1.0