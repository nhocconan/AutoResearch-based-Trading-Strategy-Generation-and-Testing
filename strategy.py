#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index + Volume Spike + Previous Day Range Breakout
# Uses Choppiness Index (14) on 12h to filter range (CHOP > 61.8) vs trend (CHOP < 38.2).
# In ranging markets: mean-reversion at previous day's high/low with volume spike.
# In trending markets: breakout of previous day's high/low with volume spike.
# Volume confirmation: current volume > 2.0 * 20-period median volume.
# Works in bull (breakouts up) and bear (breakouts down) markets.
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for previous day's high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Load 12h data for Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous day's high and low (shifted by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Align previous day's high/low to 4h timeframe
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Calculate True Range for Choppiness Index
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate +DI and -DI for ADX component of Choppiness
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (14-period)
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
    
    # Calculate DX and ADX (14-period)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index: CHOP = 100 * log10(SUM(TR,14) / (ATR(14)*14)) / log10(14)
    atr_14 = tr_smooth  # Already smoothed TR = ATR
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14 + 1e-10)) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_high_1d_aligned[i]) or np.isnan(prev_low_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            continue
        
        # Volume spike condition: current volume > 2.0 * 20-period median
        vol_median = np.median(volume[max(0, i-19):i+1])
        volume_spike = volume[i] > 2.0 * vol_median
        
        # Range market: CHOP > 61.8 -> mean reversion at previous day levels
        if chop_aligned[i] > 61.8:
            # Long: price near previous day low + volume spike
            if (close[i] <= prev_low_1d_aligned[i] * 1.005 and  # Within 0.5% of low
                volume_spike and
                position <= 0):
                position = 1
                signals[i] = base_size
            
            # Short: price near previous day high + volume spike
            elif (close[i] >= prev_high_1d_aligned[i] * 0.995 and  # Within 0.5% of high
                  volume_spike and
                  position >= 0):
                position = -1
                signals[i] = -base_size
        
        # Trending market: CHOP < 38.2 -> breakout of previous day levels
        elif chop_aligned[i] < 38.2:
            # Long: breakout above previous day high + volume spike
            if (close[i] > prev_high_1d_aligned[i] and
                volume_spike and
                position <= 0):
                position = 1
                signals[i] = base_size
            
            # Short: breakout below previous day low + volume spike
            elif (close[i] < prev_low_1d_aligned[i] and
                  volume_spike and
                  position >= 0):
                position = -1
                signals[i] = -base_size
        
        # Exit conditions: opposite signal or CHOP in middle range (40-60)
        elif position == 1 and (close[i] < prev_low_1d_aligned[i] * 0.995 or
                                40 <= chop_aligned[i] <= 60):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > prev_high_1d_aligned[i] * 1.005 or
                                 40 <= chop_aligned[i] <= 60):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Chop_Volume_Pivot_MR_Trend"
timeframe = "4h"
leverage = 1.0