#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses Donchian channel (20-bar high/low) for breakout detection, filtered by weekly pivot
# direction (price above/below weekly pivot point) and volume spike (>2.0x 20-bar avg volume).
# Weekly pivot provides institutional reference point; Donchian captures momentum; volume
# confirms participation. Designed for 6h timeframe to achieve 12-37 trades/year.
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for weekly pivot point (P = (H+L+C)/3)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: P = (H+L+C)/3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe (using previous week's pivot)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # 6h Donchian channel (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 40  # Need 20 for Donchian + buffer
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Donchian breakout conditions
        breakout_up = curr_close > high_ma_20[i]   # Break above 20-period high
        breakout_down = curr_close < low_ma_20[i]  # Break below 20-period low
        
        # Weekly pivot direction filter
        above_pivot = curr_close > weekly_pivot_aligned[i]
        below_pivot = curr_close < weekly_pivot_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up, above weekly pivot, volume spike
            if breakout_up and above_pivot and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down, below weekly pivot, volume spike
            elif breakout_down and below_pivot and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown or pivot rejection
            if curr_close < low_ma_20[i] or curr_close < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout or pivot rejection
            if curr_close > high_ma_20[i] or curr_close > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals