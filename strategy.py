#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 4h EMA21 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper (20) AND 4h EMA21 rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower (20) AND 4h EMA21 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the Donchian channel.
# Trend filter ensures directional alignment, volume confirms institutional participation.
# Target: 150-250 total trades over 4 years (38-63/year) with controlled risk.

name = "4h_Donchian_20_4hEMA21_Volume"
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
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA21 for trend filter
    ema21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # 4h EMA21 direction
    ema21_rising = np.zeros_like(ema21_4h_aligned, dtype=bool)
    ema21_falling = np.zeros_like(ema21_4h_aligned, dtype=bool)
    ema21_rising[1:] = ema21_4h_aligned[1:] > ema21_4h_aligned[:-1]
    ema21_falling[1:] = ema21_4h_aligned[1:] < ema21_4h_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average (on 4h timeframe)
    vol_ma20_4h = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma20_4h)
    volume_filter = df_4h['volume'].values > (1.5 * vol_ma20_4h_aligned)
    volume_filter_aligned = align_htf_to_ltf(prices, df_4h, volume_filter)
    
    # Donchian(20) on 4h timeframe
    high_20 = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 2)  # Sufficient warmup for EMA21 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema21_4h_aligned[i]) or np.isnan(ema21_rising[i]) or 
            np.isnan(ema21_falling[i]) or np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, EMA21 rising, volume filter
            long_cond = (close[i] > upper_20_aligned[i]) and ema21_rising[i] and volume_filter_aligned[i]
            # Short conditions: price breaks below lower Donchian, EMA21 falling, volume filter
            short_cond = (close[i] < lower_20_aligned[i]) and ema21_falling[i] and volume_filter_aligned[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below lower Donchian (mean reversion)
            if close[i] < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above upper Donchian (mean reversion)
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals