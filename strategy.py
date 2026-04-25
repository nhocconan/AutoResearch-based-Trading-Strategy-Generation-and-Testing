#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_VolumeSpike
Hypothesis: Weekly pivot points (calculated from prior week's OHLC) provide key support/resistance levels on 6h timeframe. Donchian(20) breakouts in direction of weekly pivot bias (price above/below weekly pivot) with volume confirmation (>2.0x 20-bar average) capture institutional participation. Works in bull markets via long breakouts above weekly pivot and bear markets via short breakouts below weekly pivot. Weekly pivot adds structural edge vs daily pivot alone. Target 12-37 trades/year to minimize fee drag. Uses discrete position sizing (0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points from prior completed week
    # Weekly pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # Use previous completed 1w bar to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close_1w[:-1]])
    prev_high = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low = np.concatenate([[np.nan], low_1w[:-1]])
    
    weekly_pivot = (prev_high + prev_low + prev_close) / 3.0
    weekly_range = prev_high - prev_low
    r1 = 2 * weekly_pivot - prev_low
    s1 = 2 * weekly_pivot - prev_high
    r2 = weekly_pivot + weekly_range
    s2 = weekly_pivot - weekly_range
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Donchian(20) channels on 6h
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(donchian_window, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        pivot_val = weekly_pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        dch_high = donchian_high[i]
        dch_low = donchian_low[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Donchian breakout with weekly pivot bias and volume
            # Long: price breaks above Donchian high AND above weekly pivot (bullish bias) with volume spike
            long_signal = (high_val > dch_high) and (close_val > pivot_val) and volume_spike
            # Short: price breaks below Donchian low AND below weekly pivot (bearish bias) with volume spike
            short_signal = (low_val < dch_low) and (close_val < pivot_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price breaks below weekly pivot (bias change)
            if close_val < pivot_val:
                signals[i] = 0.0
                position = 0
            # 2. Opposite Donchian breakout (strong reversal)
            elif low_val < dch_low:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price breaks above weekly pivot (bias change)
            if close_val > pivot_val:
                signals[i] = 0.0
                position = 0
            # 2. Opposite Donchian breakout (strong reversal)
            elif high_val > dch_high:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0