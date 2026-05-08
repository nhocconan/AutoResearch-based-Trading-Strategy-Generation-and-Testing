#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-day Donchian high AND weekly price > weekly SMA(50) AND volume > 1.5x 20-day average volume.
# Short when price breaks below 20-day Donchian low AND weekly price < weekly SMA(50) AND volume > 1.5x 20-day average volume.
# Exit when price crosses back inside the Donchian channel (between 20-day low and high).
# Uses 1d timeframe as specified, with 1w trend filter for higher timeframe context.
# Target: 30-100 total trades over 4 years (7-25/year) with controlled frequency to avoid fee drag.

name = "1d_Donchian_20_WeeklyTrend_Volume"
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
    
    # Daily data for Donchian channel calculation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day Donchian high and low
    donchian_high = pd.Series(df_d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe (already aligned but using for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_d, donchian_low)
    
    # Weekly data for trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # Calculate 50-week SMA
    weekly_sma50 = pd.Series(df_w['close']).rolling(window=50, min_periods=50).mean().values
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_w, weekly_sma50)
    
    # Daily volume filter: current volume > 1.5x 20-day average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Sufficient warmup for Donchian and weekly SMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_sma50_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, weekly uptrend, volume filter
            long_cond = (close[i] > donchian_high_aligned[i]) and \
                        (close[i] > weekly_sma50_aligned[i]) and \
                        volume_filter[i]
            # Short conditions: price breaks below Donchian low, weekly downtrend, volume filter
            short_cond = (close[i] < donchian_low_aligned[i]) and \
                         (close[i] < weekly_sma50_aligned[i]) and \
                         volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals