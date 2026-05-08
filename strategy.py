#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-hour Donchian breakout with volume confirmation and 1-day trend filter.
# Long when price breaks above 4h Donchian high (20) AND volume > 1.5x 20-period average AND 1d EMA50 uptrend.
# Short when price breaks below 4h Donchian low (20) AND volume > 1.5x 20-period average AND 1d EMA50 downtrend.
# Exit when price crosses back inside the 4h Donchian channel.
# Uses 4h for signal direction, 1h for entry timing, 1d for trend filter.
# Target: 60-150 total trades over 4 years (15-37/year) with controlled frequency to avoid fee drag.

name = "1h_Donchian20_Volume_1dTrend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian high/low (20-period)
    donch_high = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 4h Donchian high, volume filter, 1d EMA50 uptrend
            long_cond = (close[i] > donch_high_aligned[i]) and volume_filter[i] and (close[i] > ema_50_aligned[i])
            # Short conditions: price breaks below 4h Donchian low, volume filter, 1d EMA50 downtrend
            short_cond = (close[i] < donch_low_aligned[i]) and volume_filter[i] and (close[i] < ema_50_aligned[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses back below 4h Donchian low
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses back above 4h Donchian high
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals