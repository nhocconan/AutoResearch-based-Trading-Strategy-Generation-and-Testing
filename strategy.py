#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyPivot_Direction_Volume
Hypothesis: 6h Donchian(20) breakouts in the direction of the weekly pivot (above/below weekly pivot from previous week) 
with volume confirmation produce reliable trends. Weekly pivot acts as a trend filter: only take long breakouts 
when above weekly pivot, short breakouts when below. This avoids counter-trend trades in ranging markets. 
Volume confirmation ensures breakout strength. Designed for 6s timeframe to capture medium-term moves with 
lower frequency to avoid fee drag. Works in both bull (breakouts continue) and bear (breakdowns continue) markets 
by following the weekly pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (standard: (H+L+C)/3) from previous week
    pivot_1w = np.full_like(high_1w, np.nan)
    for i in range(1, len(close_1w)):
        pivot_1w[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
    
    # Get daily data for volume average (more stable)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on daily timeframe
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    vol_period = 20
    if len(volume_1d) >= vol_period:
        for i in range(vol_period, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i - vol_period:i])
    
    # Donchian channels (20-period) on 6h data
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    lookback = 20
    for i in range(lookback, len(high)):
        highest_high[i] = np.max(high[i - lookback:i])
        lowest_low[i] = np.min(low[i - lookback:i])
    
    # Align weekly pivot and daily volume MA to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    vol_ma_6h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_6h[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-day average volume (scaled)
        # Approximate: 1 day ≈ 4 six-hour bars, so scale daily MA
        vol_ma_scaled = vol_ma_6h[i] / 4.0  # rough 6h average from daily
        vol_confirm = volume[i] > 1.5 * vol_ma_scaled
        
        if position == 0:
            # Long: breakout above Donchian high AND above weekly pivot
            if close[i] > highest_high[i] and close[i] > pivot_6h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low AND below weekly pivot
            elif close[i] < lowest_low[i] and close[i] < pivot_6h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below Donchian low OR price crosses below weekly pivot
            if close[i] < lowest_low[i] or close[i] < pivot_6h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above Donchian high OR price crosses above weekly pivot
            if close[i] > highest_high[i] or close[i] > pivot_6h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0