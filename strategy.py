#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_Filter_v1
Hypothesis: On 6h timeframe, buy breakouts above 20-period Donchian high when 1d EMA50 is rising, sell breakdowns below 20-period Donchian low when 1d EMA50 is falling. Uses 1d trend filter to avoid counter-trend trades, targeting 12-30 trades/year. Works in bull markets via long breakouts and in bear markets via short breakdowns, with trend alignment reducing whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian, 50 for 1d EMA
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout with 1d trend confirmation
            # Long: break above Donchian high + 1d EMA50 rising
            long_entry = (close_val > highest_high[i]) and (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1])
            # Short: break below Donchian low + 1d EMA50 falling
            short_entry = (close_val < lowest_low[i]) and (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1])
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price retouches Donchian low or 1d EMA50 turns down
            if (close_val < lowest_low[i]) or (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price retouches Donchian high or 1d EMA50 turns up
            if (close_val > highest_high[i]) or (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0