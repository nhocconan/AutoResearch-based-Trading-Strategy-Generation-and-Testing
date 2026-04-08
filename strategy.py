#!/usr/bin/env python3
# 6h_donchian20_daily_pivot_volume_v1
# Hypothesis: 6h Donchian(20) breakout with daily pivot directional bias and volume confirmation.
# Long when price breaks above Donchian(20) high, daily pivot > prior day close, and volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low, daily pivot < prior day close, and volume > 1.5x average.
# Exit when price re-enters Donchian channel or volume drops below average.
# Uses daily pivot for bias to work in both bull/bear regimes. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_daily_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian(20) channels
    donchian_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donchian_len - 1, n):
        upper[i] = np.max(high[i-donchian_len+1:i+1])
        lower[i] = np.min(low[i-donchian_len+1:i+1])
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period - 1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 1d data for daily pivot
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot point: (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Pivot bias: 1 if today's pivot > yesterday's close, -1 if <, 0 otherwise
    pivot_bias_1d = np.full(len(close_1d), 0)
    for i in range(1, len(close_1d)):
        if pivot_1d[i] > close_1d[i-1]:
            pivot_bias_1d[i] = 1
        elif pivot_1d[i] < close_1d[i-1]:
            pivot_bias_1d[i] = -1
    
    # Align pivot bias to 6h timeframe
    pivot_bias_aligned = align_htf_to_ltf(prices, df_1d, pivot_bias_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donchian_len, vol_ma_period, 1) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]) or np.isnan(pivot_bias_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below Donchian upper or volume drops below average
            if close[i] < upper[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above Donchian lower or volume drops below average
            if close[i] > lower[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian upper, pivot bias up, volume surge
            if (close[i] > upper[i] and 
                pivot_bias_aligned[i] > 0 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower, pivot bias down, volume surge
            elif (close[i] < lower[i] and 
                  pivot_bias_aligned[i] < 0 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals