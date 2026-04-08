#!/usr/bin/env python3
# 12h_donchian20_daily_pivot_volume_v1
# Hypothesis: Breakout from daily Donchian channels with pivot level confirmation and volume filter.
# Long when: price breaks above daily Donchian high(20) AND price > daily pivot point AND volume > 1.5x average.
# Short when: price breaks below daily Donchian low(20) AND price < daily pivot point AND volume > 1.5x average.
# Exit when price returns to pivot point or volume drops below average.
# Uses daily timeframe for structure (Donchian/pivot) and 12h for execution to avoid overtrading.
# Target: 15-30 trades/year.

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
    
    # Get daily data for Donchian channels and pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Donchian channels (20-period)
    donchian_period = 20
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Daily pivot point: (H + L + C) / 3
    pivot_point = (high_1d + low_1d + close_1d) / 3.0
    
    # Align daily levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    pivot_point_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donchian_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(pivot_point_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below pivot point or volume drops below average
            if close[i] < pivot_point_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above pivot point or volume drops below average
            if close[i] > pivot_point_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian high AND price > pivot AND volume surge
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > pivot_point_aligned[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low AND price < pivot AND volume surge
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < pivot_point_aligned[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals