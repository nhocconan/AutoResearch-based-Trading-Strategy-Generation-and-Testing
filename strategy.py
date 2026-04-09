#!/usr/bin/env python3
# 1d_1w_donchian_breakout_v1
# Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-day high with weekly price above weekly SMA50 and volume > 1.5x 20-day average.
# Short when price breaks below 20-day low with weekly price below weekly SMA50 and volume > 1.5x 20-day average.
# Exit when price returns to opposite Donchian level (20-day low for longs, 20-day high for shorts).
# Position size fixed at 0.25 to limit drawdown. Target: 30-100 total trades over 4 years (7-25/year).
# Works in bull markets via breakout continuation and in bear markets via mean reversion at extreme levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_v1"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMA50 for trend filter
    close_1w = df_1w['close'].values
    sma_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        for i in range(50, len(close_1w)):
            sma_50_1w[i] = np.mean(close_1w[i-50:i])
    
    # Align weekly SMA50 to daily timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate 20-day Donchian channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 20-day average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(sma_50_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below 20-day low
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above 20-day high
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 20-day high with trend and volume filters
            if (close[i] > donchian_high[i] and 
                close[i] > sma_50_1w_aligned[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 20-day low with trend and volume filters
            elif (close[i] < donchian_low[i] and 
                  close[i] < sma_50_1w_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals