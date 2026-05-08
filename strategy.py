#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and ADX trend filter.
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND ADX(14) > 25.
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND ADX(14) > 25.
# Exit when price crosses back below Donchian(10) high (for long) or above Donchian(10) low (for short).
# Uses Donchian for trend breakouts with volume and trend confirmation to avoid false breakouts.
# Target: 80-150 total trades over 4 years (20-38/year) for low fee drift.

name = "4h_Donchian20_1dVolume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Donchian channels (10-period for exit)
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # 1d volume filter: current volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = vol_1d > (1.5 * vol_ma20_1d_aligned)
    
    # Calculate ADX(14) on 4h data
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.insert(tr, 0, 0)
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian(20) high, volume filter, ADX > 25
            long_cond = (close[i] > highest_high[i]) and volume_filter[i] and (adx[i] > 25)
            # Short conditions: price breaks below Donchian(20) low, volume filter, ADX > 25
            short_cond = (close[i] < lowest_low[i]) and volume_filter[i] and (adx[i] > 25)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian(10) high
            if close[i] < highest_high_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian(10) low
            if close[i] > lowest_low_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals