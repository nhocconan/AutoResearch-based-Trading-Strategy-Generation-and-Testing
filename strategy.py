#!/usr/bin/env python3
# 6d_donchian_breakout_weekly_pivot_volume_v1
# Hypothesis: Daily Donchian breakout with weekly pivot direction filter and volume confirmation.
# Uses daily timeframe for breakout signals, weekly pivot for trend direction, and volume for confirmation.
# Works in both bull and bear markets by following higher timeframe trend while capturing breakouts.
# Target: 15-30 trades/year via strict breakout conditions + trend alignment + volume filter.

name = "6d_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period average volume for volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    r2 = pivot + (high_weekly - low_weekly)
    s2 = pivot - (high_weekly - low_weekly)
    r3 = high_weekly + 2 * (pivot - low_weekly)
    s3 = low_weekly - 2 * (high_weekly - pivot)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(20, 20)  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Get aligned weekly pivot values
        pivot_val = align_htf_to_ltf(prices, df_weekly, pivot)[i]
        r1_val = align_htf_to_ltf(prices, df_weekly, r1)[i]
        s1_val = align_htf_to_ltf(prices, df_weekly, s1)[i]
        r2_val = align_htf_to_ltf(prices, df_weekly, r2)[i]
        s2_val = align_htf_to_ltf(prices, df_weekly, s2)[i]
        r3_val = align_htf_to_ltf(prices, df_weekly, r3)[i]
        s3_val = align_htf_to_ltf(prices, df_weekly, s3)[i]
        
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(pivot_val) or np.isnan(r1_val) or np.isnan(s1_val) or
            np.isnan(r2_val) or np.isnan(s2_val) or np.isnan(r3_val) or np.isnan(s3_val) or
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price closes below Donchian lower OR price breaks below S1
            if close[i] < low_20[i] or close[i] < s1_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price closes above Donchian upper OR price breaks above R1
            if close[i] > high_20[i] or close[i] > r1_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout: price breaks above Donchian upper with volume, above weekly pivot
            if (close[i] > high_20[i] and 
                volume_filter and 
                close[i] > pivot_val):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below Donchian lower with volume, below weekly pivot
            elif (close[i] < low_20[i] and 
                  volume_filter and 
                  close[i] < pivot_val):
                position = -1
                signals[i] = -0.25
    
    return signals