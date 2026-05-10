#!/usr/bin/env python3
"""
4h_PriceChannel_Volume_Trend
Hypothesis: Price channel breakouts (Donchian 20) aligned with 1-day EMA34 trend and volume spikes yield high-probability trades.
Works in bull markets via buying breakouts above resistance and in bear markets via selling breakdowns below support.
Uses 1-day EMA34 for trend filter, Donchian(20) for entry/exit, volume confirmation (>1.5x 20-bar average) to avoid false signals.
Designed for low trade frequency (target: 20-50 trades per year) to minimize fee drag.
"""

name = "4h_PriceChannel_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after Donchian warmup (20) and EMA34 warmup (34)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + uptrend + volume
            if close[i] > highest_high[i] and uptrend and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + downtrend + volume
            elif close[i] < lowest_low[i] and downtrend and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend fails
            if close[i] < lowest_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend fails
            if close[i] > highest_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals