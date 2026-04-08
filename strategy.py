#!/usr/bin/env python3
# 12h_donchian20_daily_pivot_volume_v2
# Hypothesis: 12h Donchian(20) breakout with 1d pivot level confirmation and volume filter.
# Long when price breaks above upper Donchian channel and near 1d pivot/resistance levels with volume surge.
# Short when price breaks below lower Donchian channel and near 1d pivot/support levels with volume surge.
# Uses pivot levels as dynamic support/resistance to filter breakouts, reducing false signals.
# Volume surge confirms institutional interest. Target: 15-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_daily_pivot_volume_v2"
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
    
    # 12h Donchian channel (20-period)
    donch_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donch_len-1, n):
        upper[i] = np.max(high[i-donch_len+1:i+1])
        lower[i] = np.min(low[i-donch_len+1:i+1])
    
    # Volume filter: 2.0x 20-period average (higher threshold for fewer trades)
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 2.0 * vol_ma[i]
    
    # Get 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate classic pivot points: P = (H+L+C)/3
    # Support 1 = 2*P - H, Resistance 1 = 2*P - L
    pivot = np.full(len(close_1d), np.nan)
    r1 = np.full(len(close_1d), np.nan)
    s1 = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(high_1d[i]) and not np.isnan(low_1d[i]) and not np.isnan(close_1d[i]):
            pivot[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
            r1[i] = 2 * pivot[i] - high_1d[i]
            s1[i] = 2 * pivot[i] - low_1d[i]
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donch_len, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below lower Donchian or volume drops below average
            if close[i] < lower[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above upper Donchian or volume drops below average
            if close[i] > upper[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above upper Donchian, near pivot/R1 support, volume surge
            # Price must be within 1% of pivot or R1 to consider it a valid bounce level
            near_pivot_or_r1 = (abs(close[i] - pivot_aligned[i]) / pivot_aligned[i] < 0.01 or 
                               abs(close[i] - r1_aligned[i]) / r1_aligned[i] < 0.01)
            if (close[i] > upper[i] and 
                near_pivot_or_r1 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower Donchian, near pivot/S1 resistance, volume surge
            elif (close[i] < lower[i] and 
                  (abs(close[i] - pivot_aligned[i]) / pivot_aligned[i] < 0.01 or 
                   abs(close[i] - s1_aligned[i]) / s1_aligned[i] < 0.01) and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals