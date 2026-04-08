#!/usr/bin/env python3
# 1d_1w_donchian20_volume_sma_filter_v1
# Hypothesis: On 1d timeframe, breakouts above 20-day Donchian channel with volume > 1.5x 20-day average SMA.
# Weekly trend filter: price must be above weekly SMA(50) for longs, below for shorts.
# This filters out counter-trend moves and reduces false breakouts in choppy markets.
# Volume confirmation ensures breakouts have conviction.
# Target: 7-25 trades/year (~30-100 total over 4 years) to stay within fee-efficient range.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian20_volume_sma_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 1d
    donchian_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper_channel[i] = np.max(high[i-donchian_period+1:i+1])
        lower_channel[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Volume filter: 1.5x 20-period average SMA
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period - 1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get weekly data for trend filter (SMA 50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_sma_period = 50
    weekly_sma = np.full(len(weekly_close), np.nan)
    
    for i in range(weekly_sma_period - 1, len(weekly_close)):
        weekly_sma[i] = np.mean(weekly_close[i-weekly_sma_period+1:i+1])
    
    # Align weekly SMA to daily timeframe (wait for weekly bar to close)
    weekly_sma_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donchian_period, vol_ma_period, weekly_sma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_sma_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below lower Donchian channel
            if close[i] < lower_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above upper Donchian channel
            if close[i] > upper_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above upper Donchian + volume surge + above weekly SMA
            if (close[i] > upper_channel[i] and vol_surge[i] and 
                close[i] > weekly_sma_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below lower Donchian + volume surge + below weekly SMA
            elif (close[i] < lower_channel[i] and vol_surge[i] and 
                  close[i] < weekly_sma_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals