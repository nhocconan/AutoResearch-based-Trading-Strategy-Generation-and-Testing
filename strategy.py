#!/usr/bin/env python3
# 4h_triple_filter_breakout_v1
# Hypothesis: 4h Donchian breakout with volume confirmation and 1-day EMA trend filter for both bull and bear markets.
# Uses only 3 conditions: price breakout above/below 20-period Donchian channel, volume > 1.5x 20-period average, and price above/below 1-day EMA50 for trend alignment.
# Target: 20-30 trades/year (80-120 over 4 years) with low frequency to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_triple_filter_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period Donchian channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:
            high_max[i] = np.max(high[i-19:i+1])
            low_min[i] = np.min(low[i-19:i+1])
    
    donchian_high[19:] = high_max[19:]
    donchian_low[19:] = low_min[19:]
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA50 on daily
    ema50_1d = np.full(len(df_1d), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(df_1d)):
        if i == 0:
            ema50_1d[i] = close_1d[i]
        elif np.isnan(ema50_1d[i-1]):
            ema50_1d[i] = close_1d[i]
        else:
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Align daily EMA50 to 4h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend changes
            if close[i] < donchian_low[i] or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend changes
            if close[i] > donchian_high[i] or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high + volume + above daily EMA50
            if (close[i] > donchian_high[i] and 
                vol_ok and 
                close[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low + volume + below daily EMA50
            elif (close[i] < donchian_low[i] and 
                  vol_ok and 
                  close[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals