#!/usr/bin/env python3
"""
6h Donchian breakout with volume confirmation and 1d trend filter
Hypothesis: Donchian channel breakouts capture breakout moves, volume confirms institutional participation,
and 1d EMA200 filters for trend direction to avoid counter-trend trades. Works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_prev = np.roll(ema200_1d, 1)
    ema200_1d_prev[0] = ema200_1d[0]
    ema200_up = ema200_1d > ema200_1d_prev
    ema200_down = ema200_1d < ema200_1d_prev
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    ema200_up_aligned = align_htf_to_ltf(prices, df_1d, ema200_up)
    ema200_down_aligned = align_htf_to_ltf(prices, df_1d, ema200_down)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For EMA200 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(ema200_up_aligned[i]) or np.isnan(ema200_down_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite breakout or stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + trend
            bull_breakout = (close[i] > donchian_high[i] and 
                            volume[i] > vol_ema[i] * 1.5 and 
                            ema200_up_aligned[i])
            bear_breakout = (close[i] < donchian_low[i] and 
                            volume[i] > vol_ema[i] * 1.5 and 
                            ema200_down_aligned[i])
            
            if bull_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals