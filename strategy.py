#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: Uses 12h Donchian breakout (20-period) with 1d trend filter (price > SMA50) and volume confirmation (volume > 1.5x 20-period average).
# Works in bull via breakout continuation and bear via short breakdowns. Volume ensures breakout validity, trend filter avoids counter-trend trades.
# Designed for 15-30 trades/year on 12h to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 12-period SMA for volume average
    volume_sma = np.zeros(n)
    volume_sum = 0.0
    for i in range(n):
        volume_sum += volume[i]
        if i >= 20:
            volume_sum -= volume[i-20]
        if i < 19:
            volume_sma[i] = np.nan
        else:
            volume_sma[i] = volume_sum / 20.0
    
    # Donchian channels (20-period) on 12h
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            continue
        start_idx = max(0, i-19)
        high_max[i] = np.max(high[start_idx:i+1])
        low_min[i] = np.min(low[start_idx:i+1])
    
    # 1-day SMA50 trend filter
    sma50_1d = np.full(len(close_1d), np.nan)
    sma_sum = 0.0
    for i in range(len(close_1d)):
        sma_sum += close_1d[i]
        if i >= 49:
            sma_sum -= close_1d[i-49]
        if i < 49:
            sma50_1d[i] = np.nan
        else:
            sma50_1d[i] = sma_sum / 50.0
    
    # Align 1d SMA50 to 12h
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 49  # Ensure SMA50 is ready
    
    for i in range(start_idx, n):
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(sma50_1d_aligned[i]) or np.isnan(volume_sma[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * volume_sma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian or trend fails
            if close[i] < low_min[i] or close[i] <= sma50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian or trend fails
            if close[i] > high_max[i] or close[i] >= sma50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian with volume and uptrend
            if close[i] > high_max[i] and vol_confirm and close[i] > sma50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian with volume and downtrend
            elif close[i] < low_min[i] and vol_confirm and close[i] < sma50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals