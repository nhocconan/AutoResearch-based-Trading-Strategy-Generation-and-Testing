#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND price > 1d EMA50 AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band AND price < 1d EMA50 AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the Donchian channel.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled frequency to avoid fee drag.

name = "4h_Donchian_20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for EMA trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema50_d = pd.Series(df_d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_d_aligned = align_htf_to_ltf(prices, df_d, ema50_d)
    
    # Donchian channel (20-period) on 4h data
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for Donchian and volume
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema50_d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, above 1d EMA50, volume filter
            long_cond = (close[i] > donchian_upper[i]) and (close[i] > ema50_d_aligned[i]) and volume_filter[i]
            # Short conditions: price breaks below Donchian lower, below 1d EMA50, volume filter
            short_cond = (close[i] < donchian_lower[i]) and (close[i] < ema50_d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals