#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_PriceChannel_VolumeBreakout_1wTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for long-term trend
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily average volume (20-period)
    vol_1d = df_1d['volume'].values
    avg_vol_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    # 12h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for weekly EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(avg_vol_20_aligned[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume spike + price above weekly EMA200
            long_cond = (close[i] > high_max[i]) and (volume[i] > 1.5 * avg_vol_20_aligned[i]) and (close[i] > ema200_1w_aligned[i])
            
            # Short: price breaks below lower Donchian + volume spike + price below weekly EMA200
            short_cond = (close[i] < low_min[i]) and (volume[i] > 1.5 * avg_vol_20_aligned[i]) and (close[i] < ema200_1w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below lower Donchian
            if close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper Donchian
            if close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Donchian breakout with volume confirmation and weekly trend filter.
# Long when price breaks above 20-period high with volume >1.5x daily average and price above weekly EMA200.
# Short when price breaks below 20-period low with volume spike and price below weekly EMA200.
# Exit on opposite Donchian break. Weekly EMA200 filter ensures trading with long-term trend.
# Volume spike confirms institutional interest. Target: 50-150 total trades over 4 years.