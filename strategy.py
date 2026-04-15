#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + volume confirmation + 1d Choppiness regime filter
# Uses 12h Williams Alligator (SMA-based) for trend identification, volume to confirm trend strength,
# and 1d Choppiness Index to avoid ranging markets. Works in both bull and bear by
# only taking trades when Alligator is "awake" (trending) and price is outside Alligator teeth.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for price action and Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Alligator on 12h (all SMAs)
    # Jaw: 13-period SMA, shifted 8 bars forward
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars forward
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate Choppiness Index (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (hh_14 - ll_14 + 1e-10)) / np.log10(14)
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or
            np.isnan(lips_12h_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price above Alligator (lips > teeth > jaw) + volume spike + chop < 61.8 (trending)
        if (lips_12h_aligned[i] > teeth_12h_aligned[i] and
            teeth_12h_aligned[i] > jaw_12h_aligned[i] and
            close[i] > lips_12h_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            chop_aligned[i] < 61.8 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below Alligator (jaw > teeth > lips) + volume spike + chop < 61.8 (trending)
        elif (jaw_12h_aligned[i] > teeth_12h_aligned[i] and
              teeth_12h_aligned[i] > lips_12h_aligned[i] and
              close[i] < jaw_12h_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              chop_aligned[i] < 61.8 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or chop > 61.8 (ranging market) or price re-enters Alligator mouth
        elif position == 1 and (close[i] < teeth_12h_aligned[i] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > teeth_12h_aligned[i] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_Volume_Chop_Filter"
timeframe = "12h"
leverage = 1.0