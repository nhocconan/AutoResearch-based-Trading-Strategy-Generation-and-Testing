#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_Volume
Hypothesis: Price breaks 6h Donchian(20) high/low with weekly pivot direction filter (1w trend) and volume confirmation.
Weekly pivot determines overall trend: price above weekly pivot = long bias, below = short bias.
Volume confirmation ensures breakouts are genuine. Works in bull/bear by aligning with weekly trend.
Target: 15-25 trades/year (60-100 total) to minimize fee drag.
"""

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Point (standard calculation)
    # PP = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # 6h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            highest_high[i] = np.max(high[i - lookback + 1:i + 1])
            lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Align weekly pivot to 6h
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback - 1  # Wait for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(pp_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: price above/below weekly pivot
        is_uptrend_bias = close[i] > pp_1w_aligned[i]
        is_downtrend_bias = close[i] < pp_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = high[i] > highest_high[i]
        breakout_down = low[i] < lowest_low[i]
        
        # Volume confirmation: current volume > 1.5x average of last 6 periods
        if i >= 6:
            vol_avg = np.mean(volume[i-6:i])
            volume_confirm = volume[i] > 1.5 * vol_avg
        else:
            volume_confirm = False
        
        if position == 0:
            # Long: breakout above Donchian high, weekly uptrend bias, volume
            if breakout_up and is_uptrend_bias and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low, weekly downtrend bias, volume
            elif breakout_down and is_downtrend_bias and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or trend bias shifts
            if low[i] < lowest_low[i] or not is_uptrend_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or trend bias shifts
            if high[i] > highest_high[i] or not is_downtrend_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals