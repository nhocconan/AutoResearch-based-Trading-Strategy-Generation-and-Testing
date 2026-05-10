#!/usr/bin/env python3
# 4h_Vortex_BullBear_Filter
# Hypothesis: Use Vortex Indicator on daily timeframe to filter 4h bull/bear regimes, then trade 4h breakouts of 20-bar high/low with volume confirmation.
# Vortex identifies trend strength and direction; reduces false breakouts in choppy markets. Works in both bull and bear by aligning with higher-timeframe trend.
# Targets 20-40 trades/year with discrete sizing (0.25) to minimize fee drag.

name = "4h_Vortex_BullBear_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Vortex trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Vortex Indicator on daily data
    # VM+ = |today's high - yesterday's low|
    # VM- = |today's low - yesterday's high|
    # True Range = max(|high-low|, |high-prev close|, |low-prev close|)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    tr = np.maximum(np.abs(high_1d - low_1d),
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                               np.abs(low_1d - np.roll(close_1d, 1))))
    
    # Handle first element (no previous day)
    vm_plus[0] = 0
    vm_minus[0] = 0
    tr[0] = high_1d[0] - low_1d[0]  # approximate TR for first bar
    
    # Vortex values (14-period)
    period = 14
    vm_plus_sum = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Align Vortex to 4h timeframe (Vortex needs 2 extra bars for confirmation like swing points)
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus, additional_delay_bars=2)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus, additional_delay_bars=2)
    
    # Determine trend: VI+ > VI- = uptrend, VI- > VI+ = downtrend
    # Align daily close for trend confirmation
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # 4h price channel: 20-period high/low for breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Warmup: need Vortex (14+2), price channel (20), volume MA (20)
    start_idx = 20  # dominated by 20-period lookbacks
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vi_plus_aligned[i]) or
            np.isnan(vi_minus_aligned[i]) or
            np.isnan(close_1d_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Determine trend from Vortex
        uptrend = vi_plus_aligned[i] > vi_minus_aligned[i]
        downtrend = vi_minus_aligned[i] > vi_plus_aligned[i]
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout above 20-bar high or breakdown below 20-bar low
        breakout_high = close[i] > high_20[i]
        breakdown_low = close[i] < low_20[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: Breakout above 20-bar high with volume surge and daily uptrend
            if breakout_high and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below 20-bar low with volume surge and daily downtrend
            elif breakdown_low and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            bars_since_entry += 1
            # Enforce minimum holding period of 3 bars (12 hours)
            if bars_since_entry < 3:
                signals[i] = signals[i-1]  # maintain position
                continue
            
            if position == 1:
                # Long exit: price breaks below 20-bar low or trend changes to downtrend
                if close[i] < low_20[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price breaks above 20-bar high or trend changes to uptrend
                if close[i] > high_20[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals