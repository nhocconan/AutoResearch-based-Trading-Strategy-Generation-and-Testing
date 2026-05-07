#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band (20-period) AND price > 1d EMA50 AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band (20-period) AND price < 1d EMA50 AND volume > 1.5x 20-period average.
# Exit when price crosses back below Donchian middle (10-period) for long or above middle for short.
# Designed for 4h timeframe with moderate trade frequency (target: 20-50/year) to avoid fee drag.
# Uses 1d EMA50 for trend filter to avoid counter-trend trades in strong trends.
# Volume filter ensures participation and avoids low-conviction moves.
name = "4h_Donchian_1dEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period high/low, 10-period middle)
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    high_max10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_min10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_middle = (high_max10 + low_min10) / 2.0
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(high_max20[i]) or np.isnan(low_min20[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian upper band, price > 1d EMA50, volume filter
            long_cond = (close[i] > high_max20[i]) and (close[i] > ema50_1d_aligned[i]) and volume_filter[i]
            # Short conditions: price < Donchian lower band, price < 1d EMA50, volume filter
            short_cond = (close[i] < low_min20[i]) and (close[i] < ema50_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < Donchian middle
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > Donchian middle
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals