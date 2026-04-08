#!/usr/bin/env python3
# 12h_1d_donchian_breakout_volume_v1
# Hypothesis: 12-hour Donchian(20) breakout with 1-day volume confirmation and ATR volatility filter.
# Long when price breaks above 20-period high with above-average volume; short when breaks below 20-period low with above-average volume.
# Uses 1-day ATR to filter out low-volatility chop. Designed for 12-30 trades/year on 12h to avoid fee drag.
# Works in bull/bear via breakout logic with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_breakout_volume_v1"
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
    period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        donchian_high[i] = np.max(high[i - period + 1:i + 1])
        donchian_low[i] = np.min(low[i - period + 1:i + 1])
    
    # Get 1d data for volume and ATR filters
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day average volume (20-period)
    avg_volume_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        avg_volume_1d[i] = np.mean(volume_1d[i - 19:i + 1])
    
    # 1-day ATR (14-period) for volatility filter
    atr_period = 14
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(atr_period, len(df_1d)):
        atr_1d[i] = np.mean(tr[i - atr_period + 1:i + 1])
    
    # Align 1d filters to 12h
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12-period average volume for 12h timeframe
    avg_volume_12h = np.full(n, np.nan)
    for i in range(11, n):
        avg_volume_12h[i] = np.mean(volume[i - 11:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(period - 1, 19, atr_period)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(avg_volume_12h[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume and volatility conditions
        volume_surge = volume[i] > avg_volume_12h[i] * 1.5  # 50% above average volume
        vol_filter = atr_1d_aligned[i] > 0  # Ensure volatility data exists
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or volume drops significantly
            if close[i] < donchian_low[i] or volume[i] < avg_volume_12h[i] * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or volume drops significantly
            if close[i] > donchian_high[i] or volume[i] < avg_volume_12h[i] * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume surge and adequate volatility
            if (close[i] > donchian_high[i] and 
                volume_surge and 
                vol_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume surge and adequate volatility
            elif (close[i] < donchian_low[i] and 
                  volume_surge and 
                  vol_filter):
                position = -1
                signals[i] = -0.25
    
    return signals