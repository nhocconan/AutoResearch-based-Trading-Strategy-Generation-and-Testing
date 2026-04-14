#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction (from 1w high/low) and volume confirmation.
# Weekly pivot (high/low of prior week) defines macro bias: long if price > weekly high, short if price < weekly low.
# 6h Donchian breakout in direction of weekly pivot captures momentum with trend alignment.
# Volume > 1.8x average confirms institutional participation and reduces false breakouts.
# Works in bull/bear as weekly pivot adapts to longer-term trend.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly high and low from completed weekly bar (no look-ahead)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Donchian channel (20 periods) on 6h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 1.8x average volume (higher threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len, 30)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(weekly_high_aligned[i]) or
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly pivot bias: price relative to weekly high/low
        above_weekly_high = close[i] > weekly_high_aligned[i]
        below_weekly_low = close[i] < weekly_low_aligned[i]
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirmed = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + price > weekly high + volume
            if (close[i] > dc_upper[i] and 
                above_weekly_high and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + price < weekly low + volume
            elif (close[i] < dc_lower[i] and 
                  below_weekly_low and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below weekly high or breaks below Donchian lower
            if close[i] < weekly_high_aligned[i] or close[i] < dc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns above weekly low or breaks above Donchian upper
            if close[i] > weekly_low_aligned[i] or close[i] > dc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Donchian_WeeklyPivot_Volume_v1"
timeframe = "6h"
leverage = 1.0