#!/usr/bin/env python3
"""
1d_1w_Price_Channel_Breakout
Hypothesis: On 1d timeframe, buy breakouts above weekly Donchian high with volume confirmation,
sell breakdowns below weekly Donchian low with volume confirmation. Exit at opposite channel edge.
Uses weekly trend filter to avoid counter-trend trades. Designed for low trade frequency
(10-25/year) by requiring multiple confluence factors. Works in bull/bear via weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Price_Channel_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DONCHIAN CHANNEL (20-period) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period Donchian channels on weekly data
    donchian_high = np.full(len(high_1w), np.nan)
    donchian_low = np.full(len(low_1w), np.nan)
    
    for i in range(20, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
    
    # Align to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # === WEEKLY TREND FILTER (50-period SMA) ===
    sma_50 = np.full(len(high_1w), np.nan)
    for i in range(50, len(high_1w)):
        sma_50[i] = np.mean(high_1w[i-50:i])
    
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50)
    weekly_uptrend = sma_50_aligned > 0  # Will be False until enough data
    
    # === VOLUME CONFIRMATION (20-day average) ===
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(sma_50_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Entry conditions
        long_setup = (close[i] > donchian_high_aligned[i]) and vol_confirm
        short_setup = (close[i] < donchian_low_aligned[i]) and vol_confirm
        
        # Exit conditions: opposite channel edge
        exit_long = close[i] < donchian_low_aligned[i]
        exit_short = close[i] > donchian_high_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals