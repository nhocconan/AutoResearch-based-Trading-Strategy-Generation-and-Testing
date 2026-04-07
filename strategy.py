#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian Breakout + Weekly Pivot Direction + Volume Spike
# Hypothesis: Donchian breakouts capture momentum with clear structure.
# Weekly pivot (from prior week) provides directional bias: long only above weekly pivot, short only below.
# Volume spike confirms institutional participation, filtering false breakouts.
# Works in bull via longs above pivot + volume, in bear via shorts below pivot + volume.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (prior week only)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's OHLC
    # Use prior week's data to avoid look-ahead
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align to 6h: each 6h bar gets the prior week's pivot levels
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Donchian channel (20-period) on 6h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly pivot OR Donchian low breaks
            if close[i] < pivot_aligned[i] or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.28
        elif position == -1:  # Short position
            # Exit: price closes above weekly pivot OR Donchian high breaks
            if close[i] > pivot_aligned[i] or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.28
        else:  # Flat, look for entry
            if vol_ok:
                # Long: break above Donchian high AND price above weekly pivot
                if close[i] > donchian_high[i] and close[i] > pivot_aligned[i]:
                    position = 1
                    signals[i] = 0.28
                # Short: break below Donchian low AND price below weekly pivot
                elif close[i] < donchian_low[i] and close[i] < pivot_aligned[i]:
                    position = -1
                    signals[i] = -0.28
    
    return signals