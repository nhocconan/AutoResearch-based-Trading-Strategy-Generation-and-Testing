#!/usr/bin/env python3
"""
1d Donchian breakout with 1w trend filter and volume confirmation
Hypothesis: Donchian(20) breakouts capture breakout momentum. 1w EMA200 filters trend direction to avoid counter-trend trades. Volume confirms breakout strength.
Works in both bull (buy breakouts above upper band in uptrend) and bear (sell breakdowns below lower band in downtrend). Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_prev = np.roll(ema200_1w, 1)
    ema200_1w_prev[0] = ema200_1w[0]
    ema200_rising = ema200_1w > ema200_1w_prev
    ema200_falling = ema200_1w < ema200_1w_prev
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    ema200_rising_aligned = align_htf_to_ltf(prices, df_1w, ema200_rising)
    ema200_falling_aligned = align_htf_to_ltf(prices, df_1w, ema200_falling)
    
    # 1d data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For EMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ema[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(ema200_rising_aligned[i]) or 
            np.isnan(ema200_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price closes below lower Donchian band OR stoploss
            if (close[i] <= low_20[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above upper Donchian band OR stoploss
            if (close[i] >= high_20[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            long_entry = (close[i] > high_20[i] and 
                         ema200_rising_aligned[i] and 
                         volume[i] > vol_ema[i] * 2.0)
            short_entry = (close[i] < low_20[i] and 
                          ema200_falling_aligned[i] and 
                          volume[i] > vol_ema[i] * 2.0)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals