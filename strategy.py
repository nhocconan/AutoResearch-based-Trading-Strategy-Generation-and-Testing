#!/usr/bin/env python3
# 12h_donchian20_daily_pivot_volume_v1
# Hypothesis: Combines 12h Donchian(20) breakouts with daily pivot point levels and volume confirmation.
# Long when: Price breaks above Donchian upper band, above daily R2 pivot level, and volume > 1.5x average.
# Short when: Price breaks below Donchian lower band, below daily S2 pivot level, and volume > 1.5x average.
# Exit when price crosses the daily pivot point (PP) level.
# Uses pivot points as strong support/resistance levels, Donchian for breakout confirmation, volume for momentum validation.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_daily_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian(20) channels
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: PP = (H+L+C)/3, R2 = PP + (H-L), S2 = PP - (H-L)
    pivot_pp = (high_1d + low_1d + close_1d) / 3.0
    pivot_range = high_1d - low_1d
    pivot_r2 = pivot_pp + pivot_range
    pivot_s2 = pivot_pp - pivot_range
    
    # Align pivot levels to 12h timeframe
    pivot_pp_aligned = align_htf_to_ltf(prices, df_1d, pivot_pp)
    pivot_r2_aligned = align_htf_to_ltf(prices, df_1d, pivot_r2)
    pivot_s2_aligned = align_htf_to_ltf(prices, df_1d, pivot_s2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donchian_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(pivot_pp_aligned[i]) or 
            np.isnan(pivot_r2_aligned[i]) or np.isnan(pivot_s2_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below daily pivot point
            if close[i] < pivot_pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above daily pivot point
            if close[i] > pivot_pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian upper band, above R2 pivot, volume surge
            if (close[i] > highest_high[i] and 
                close[i] > pivot_r2_aligned[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower band, below S2 pivot, volume surge
            elif (close[i] < lowest_low[i] and 
                  close[i] < pivot_s2_aligned[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals