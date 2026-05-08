#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with 12h volume confirmation and 1d trend filter.
# Long when price breaks above 20-period Donchian upper band AND 12h volume > 1.5x 20-period average AND price > 1d EMA50.
# Short when price breaks below 20-period Donchian lower band AND 12h volume > 1.5x 20-period average AND price < 1d EMA50.
# Exit when price crosses back below/above the Donchian middle band (10-period average of high/low).
# Uses Donchian breakouts for trend continuation with volume and trend filters to avoid false breakouts.
# Target: 80-160 total trades over 4 years (20-40/year) for balanced frequency and low fee drag.

name = "6h_Donchian20_12hVolume_1dEMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian channel (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (highest_high + lowest_low) / 2.0
    
    # 12h volume filter: current volume > 1.5x 20-period average
    vol_12h = df_12h['volume'].values
    vol_ma20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma20_12h)
    volume_12h = df_12h['volume'].values
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    volume_filter = volume_12h_aligned > (1.5 * vol_ma20_12h_aligned)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper band, volume spike, above 1d EMA50
            long_cond = (close[i] > donchian_upper[i]) and volume_filter[i] and (close[i] > ema50_1d_aligned[i])
            # Short conditions: break below lower band, volume spike, below 1d EMA50
            short_cond = (close[i] < donchian_lower[i]) and volume_filter[i] and (close[i] < ema50_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below middle band
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above middle band
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals