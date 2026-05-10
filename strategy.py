#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_VolumeFilter
Hypothesis: On 4h timeframe, use Donchian channel (20) breakout for entries, filtered by 12h EMA50 trend and 1d volume spike. Exit on opposite Donchian breakout or trend reversal. This captures strong trends while avoiding whipsaw, with low trade frequency to minimize fee drag. Works in bull/bear via trend filter.
"""

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 4h data for Donchian and price
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 12h EMA50
        uptrend_12h = close[i] > ema50_12h_aligned[i]
        downtrend_12h = close[i] < ema50_12h_aligned[i]
        
        # Volume filter: current 4h volume > 1.5x 1d 20-period MA
        volume_filter = volume[i] > vol_ma20_1d_aligned[i] * 1.5
        
        if position == 0:
            # Long: Donchian breakout above upper band in uptrend with volume
            if close[i] > highest_high[i] and uptrend_12h and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band in downtrend with volume
            elif close[i] < lowest_low[i] and downtrend_12h and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown below lower band or trend fails
            if close[i] < lowest_low[i] or not uptrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout above upper band or trend fails
            if close[i] > highest_high[i] or not downtrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals