#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (from 1w timeframe) and volume confirmation (>2.0x 20-bar avg) captures institutional breakouts with controlled trade frequency. Weekly pivot provides robust trend filter for both bull and bear markets. Donchian breakouts capture momentum, volume confirms participation, and discrete sizing (0.25) minimizes fee churn. Targets 12-37 trades/year (50-150 over 4 years) on 6h timeframe.
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
    
    # Get 1w data for weekly pivot direction (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using previous completed 1w bar)
    prev_high_1w = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low_1w = np.concatenate([[np.nan], low_1w[:-1]])
    prev_close_1w = np.concatenate([[np.nan], close_1w[:-1]])
    
    pivot_point = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    r1 = 2 * pivot_point - prev_low_1w
    s1 = 2 * pivot_point - prev_high_1w
    r2 = pivot_point + (prev_high_1w - prev_low_1w)
    s2 = pivot_point - (prev_high_1w - prev_low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Get 6h data for Donchian channels (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Donchian(20) channels from previous completed 6h bar
    # Using rolling window of 20 periods on 6h data
    high_ma_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous completed bar (avoid look-ahead)
    donchian_high = np.concatenate([[np.nan], high_ma_20[:-1]])
    donchian_low = np.concatenate([[np.nan], low_ma_20[:-1]])
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(20, 20)  # Donchian20, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
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
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        # Trend filter: price above/below weekly pivot
        uptrend = close_val > pivot_val
        downtrend = close_val < pivot_val
        
        if position == 0:
            # Look for entry signals: Donchian breakout with weekly pivot trend and volume
            # Long: price breaks above Donchian high with uptrend and volume spike
            long_signal = (high_val > donchian_high_val) and uptrend and volume_spike
            # Short: price breaks below Donchian low with downtrend and volume spike
            short_signal = (low_val < donchian_low_val) and downtrend and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below Donchian low (exit long)
            if close_val < donchian_low_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price crosses below weekly pivot (exit long)
            elif close_val < pivot_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above Donchian high (exit short)
            if close_val > donchian_high_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price crosses above weekly pivot (exit short)
            elif close_val > pivot_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike"
timeframe = "6h"
leverage = 1.0