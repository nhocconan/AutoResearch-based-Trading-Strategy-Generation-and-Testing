#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Reversal with Volume Spike and Trend Filter
# Uses daily Camarilla pivot levels (H3/L3) for mean-reversion entries.
# Long when price touches L3 with volume spike in downtrend (ADX > 25).
# Short when price touches H3 with volume spike in uptrend (ADX > 25).
# Works in ranging markets (reversions) and strong trends (pullbacks to pivot levels).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    # H3 = Close + 1.1 * (High - Low) / 2
    # L3 = Close - 1.1 * (High - Low) / 2
    range_1d = high_1d - low_1d
    H3 = close_1d + 1.1 * range_1d / 2
    L3 = close_1d - 1.1 * range_1d / 2
    
    # Shift to avoid look-ahead (use previous day's levels)
    prev_H3 = np.roll(H3, 1)
    prev_L3 = np.roll(L3, 1)
    prev_H3[0] = np.nan
    prev_L3[0] = np.nan
    
    # Align Camarilla levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, prev_H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, prev_L3)
    
    # Load 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period) on 12h
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(close_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(close_12h, 1)), 
                        np.maximum(np.roll(close_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(adx_aligned[i])):
            continue
        
        # Volume spike condition: current volume > 1.5 * median of last 20 bars
        vol_median = np.median(volume[max(0, i-20):i+1])
        volume_spike = volume[i] > 1.5 * vol_median
        
        # Long entry: price touches L3 (support) with volume spike in downtrend
        if (close[i] <= L3_aligned[i] * 1.001 and  # Allow small tolerance
            volume_spike and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price touches H3 (resistance) with volume spike in uptrend
        elif (close[i] >= H3_aligned[i] * 0.999 and  # Allow small tolerance
              volume_spike and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price crosses pivot point (mean reversion complete) or weak trend
        elif position == 1 and (close[i] >= close_1d[int(i/12)] if i >= 12 else False or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= close_1d[int(i/12)] if i >= 12 else False or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_Pivot_Volume_ADX"
timeframe = "12h"
leverage = 1.0