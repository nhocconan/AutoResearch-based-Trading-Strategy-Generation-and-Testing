#!/usr/bin/env python3
# [24964] 1d_1w_donchian_breakout_v1
# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-day high with volume > 1.5x average and weekly trend up.
# Short when price breaks below 20-day low with volume > 1.5x average and weekly trend down.
# Exit when price returns to 10-day moving average.
# Uses weekly trend from 1-week data for bias, effective in both trending and ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly trend: price above/below 20-week SMA
    close_1w = df_1w['close'].values
    sma_20_1w = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        sma_20_1w[i] = np.mean(close_1w[i-20:i])
    weekly_trend = sma_20_1w  # True if price above SMA (uptrend)
    
    # Align weekly trend to daily timeframe
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 10-day moving average for exit
    ma_10 = np.full(n, np.nan)
    for i in range(10, n):
        ma_10[i] = np.mean(close[i-10:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ma_10[i]) or np.isnan(vol_ma[i]) or np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to 10-day MA
            if price <= ma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 10-day MA
            if price >= ma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and weekly uptrend
            if price > donchian_high[i] and vol_ratio > 1.5 and weekly_trend_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and weekly downtrend
            elif price < donchian_low[i] and vol_ratio > 1.5 and not weekly_trend_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals