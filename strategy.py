#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h Pivot Direction + Volume Confirmation
Hypothesis: 6h Donchian breakouts capture medium-term momentum, filtered by 12h pivot
direction (bullish/bearish bias) to avoid counter-trend trades. Volume confirmation
ensures institutional participation. Works in bull markets (breakout continuation)
and bear markets (avoids false breakouts via pivot filter). Targets 12-30 trades/year
to minimize fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for pivot direction (loaded ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h pivot points (standard: PP = (H+L+C)/3, bias = close > PP)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_point = (high_12h + low_12h + close_12h) / 3.0
    pivot_bullish = close_12h > pivot_point  # True = bullish bias
    
    # Align pivot bias to 6h timeframe
    pivot_bullish_aligned = align_htf_to_ltf(prices, df_12h, pivot_bullish.astype(float))
    
    # 6h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after Donchian lookback and volume MA
    start_idx = max(lookback, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(pivot_bullish_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Donchian breakout conditions
        breakout_long = curr_close > highest_high[i]
        breakout_short = curr_close < lowest_low[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + 12h pivot alignment
            long_entry = breakout_long and vol_spike and (pivot_bullish_aligned[i] > 0.5)
            short_entry = breakout_short and vol_spike and (pivot_bullish_aligned[i] < 0.5)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on Donchian mid-line retrace or pivot flip to bearish
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if curr_close < donchian_mid or pivot_bullish_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on Donchian mid-line retrace or pivot flip to bullish
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if curr_close > donchian_mid or pivot_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0