#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume spike and 1w EMA50 trend filter.
# Long when price breaks above Donchian upper(20) AND 1d volume > 2.0x 24-period average AND price > 1w EMA50.
# Short when price breaks below Donchian lower(20) AND 1d volume > 2.0x 24-period average AND price < 1w EMA50.
# Exit when price crosses back below/above 1w EMA50 (trend-based exit).
# Uses Donchian for clear breakout signals, high volume threshold to avoid false breaks, and weekly EMA for trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency.

name = "12h_Donchian_20_1dVolume_1wEMA50"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 24:
        return np.zeros(n)
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d volume filter: current volume > 2.0x 24-period average
    vol_1d = df_1d['volume'].values
    vol_ma24 = pd.Series(vol_1d).rolling(window=24, min_periods=24).mean().values
    vol_1d_filter = vol_1d > (2.0 * vol_ma24)
    vol_1d_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_filter)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_1d_filter_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, volume spike, above 1w EMA50
            long_cond = (close[i] > highest_high[i]) and vol_1d_filter_aligned[i] and (close[i] > ema50_1w_aligned[i])
            # Short conditions: price breaks below Donchian lower, volume spike, below 1w EMA50
            short_cond = (close[i] < lowest_low[i]) and vol_1d_filter_aligned[i] and (close[i] < ema50_1w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1w EMA50 (trend change)
            if close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1w EMA50 (trend change)
            if close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals