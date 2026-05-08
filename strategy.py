#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Modified Donchian Breakout with 1d Volume Confirmation and 1w Trend Filter.
# Long when price breaks above 6h Donchian(20) high AND 1d volume > 1.5x 20-period average AND price > 1w EMA50.
# Short when price breaks below 6h Donchian(20) low AND 1d volume > 1.5x 20-period average AND price < 1w EMA50.
# Exit when price crosses back below/above 6h Donchian(10) mid-point to capture mean reversion in ranging markets.
# Uses Donchian breakouts for trend continuation with volume confirmation to avoid false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drift.

name = "6h_Donchian20_1dVolume_1wEMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 6h Donchian channels (20-period for breakout, 10-period for exit)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_mid_10 = (highest_high_10 + lowest_low_10) / 2.0
    
    # 1d volume filter: current volume > 1.5x 20-period average
    vol_ma20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > (1.5 * vol_ma20_1d_aligned)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(donchian_mid_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above Donchian(20) high, volume spike, above 1w EMA50
            long_cond = (close[i] > highest_high_20[i]) and volume_filter[i] and (close[i] > ema50_1w_aligned[i])
            # Short conditions: breakout below Donchian(20) low, volume spike, below 1w EMA50
            short_cond = (close[i] < lowest_low_20[i]) and volume_filter[i] and (close[i] < ema50_1w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian(10) mid-point
            if close[i] < donchian_mid_10[i] and close[i-1] >= donchian_mid_10[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian(10) mid-point
            if close[i] > donchian_mid_10[i] and close[i-1] <= donchian_mid_10[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals