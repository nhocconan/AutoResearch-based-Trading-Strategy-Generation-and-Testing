#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation.
# Uses Donchian channels (20-period) for breakout signals, confirmed by weekly pivot direction
# and volume spikes. Weekly pivot provides directional bias (above/below pivot) to avoid
# counter-trend trades. Volume spike filters for institutional participation.
# Designed to work in both bull and bear markets by following weekly trend direction.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years) to minimize fee drag.
name = "6h_Donchian_20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot and trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Weekly pivot from previous week
    high_prev = df_w['high'].shift(1).values
    low_prev = df_w['low'].shift(1).values
    close_prev = df_w['close'].shift(1).values
    pivot = (high_prev + low_prev + close_prev) / 3
    pivot_w = align_htf_to_ltf(prices, df_w, pivot)
    
    # 6x Donchian channel (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # 6x volume average for spike detection
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):  # 20-period MA
        vol_ma[i] = np.mean(volume[i-20:i])
    
    vol_spike = np.where(vol_ma > 0, volume / vol_ma, 1.0) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_w[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: price above/below weekly pivot
        above_pivot = close[i] > pivot_w[i]
        below_pivot = close[i] < pivot_w[i]
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume spike and above weekly pivot
            long_condition = (close[i] > highest_high[i]) and vol_spike[i] and above_pivot
            # Short breakdown: price breaks below Donchian low with volume spike and below weekly pivot
            short_condition = (close[i] < lowest_low[i]) and vol_spike[i] and below_pivot
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below Donchian high or weekly trend turns down
            if (close[i] < highest_high[i]) or (not above_pivot):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above Donchian low or weekly trend turns up
            if (close[i] > lowest_low[i]) or (not below_pivot):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals