#!/usr/bin/env python3
# 12h_donchian20_daily_pivot_volume_v1
# Hypothesis: Combines 12h Donchian(20) breakout with 1d pivot level support/resistance and volume confirmation.
# Long when: price breaks above Donchian high(20), price above daily pivot, volume > 2x average.
# Short when: price breaks below Donchian low(20), price below daily pivot, volume > 2x average.
# Exit when price crosses daily pivot opposite direction or volume drops below average.
# Uses pivot levels as dynamic support/resistance and Donchian for breakouts.
# Volume confirmation filters false breakouts. Target: 15-30 trades/year.

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
    
    # 12h Donchian channels (20-period)
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(donchian_period-1, n):
        donchian_high[i] = np.max(high[i-donchian_period+1:i+1])
        donchian_low[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Volume filter: 2x 24-period average (2 days of 12h bars)
    vol_ma_period = 24
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
    
    # Calculate pivot levels: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L
    pivot = (high_1d + low_1d + close_1d) / 3.0
    s1 = 2 * pivot - high_1d
    r1 = 2 * pivot - low_1d
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donchian_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(pivot_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below pivot or volume drops below average
            if close[i] < pivot_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above pivot or volume drops below average
            if close[i] > pivot_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian high, price above pivot, volume surge
            if (close[i] > donchian_high[i] and 
                close[i] > pivot_aligned[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low, price below pivot, volume surge
            elif (close[i] < donchian_low[i] and 
                  close[i] < pivot_aligned[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals