#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_v1
Hypothesis: Trade 6h Donchian(20) breakouts in direction of weekly Camarilla pivot bias.
Weekly Camarilla pivot (from 1w data) defines trend: price above weekly H3 = bull bias, below L3 = bear bias.
Only take longs when price breaks above Donchian(20) high AND weekly bias is bull.
Only take shorts when price breaks below Donchian(20) low AND weekly bias is bear.
Add volume confirmation (volume > 1.3 * 20-period average volume) to avoid false breakouts.
Target: 12-25 trades/year to minimize fee drag while capturing sustained moves.
Discrete sizing: 0.25.
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
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (using previous week's OHLC)
    # HLC of previous week: need to shift by 1 week to avoid look-ahead
    # We'll calculate pivot for each week using prior week's data
    # Since we get weekly data aligned, we can compute pivot from weekly OHLC
    # Camarilla formulas:
    # Pivot = (high + low + close) / 3
    # H3 = pivot + (high - low) * 1.1 / 4
    # L3 = pivot - (high - low) * 1.1 / 4
    # H4 = pivot + (high - low) * 1.1 / 2
    # L4 = pivot - (high - low) * 1.1 / 2
    
    # We need to use the COMPLETED weekly bar's OHLC to avoid look-ahead
    # Since df_1w contains historical weekly data, we can compute pivot for each week
    # and align it to 6h timeframe
    
    # Calculate pivot levels for each weekly bar
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    pivot = (wk_high + wk_low + wk_close) / 3.0
    rng = wk_high - wk_low
    H3 = pivot + rng * 1.1 / 4.0
    L3 = pivot - rng * 1.1 / 4.0
    H4 = pivot + rng * 1.1 / 2.0
    L4 = pivot - rng * 1.1 / 2.0
    
    # Align weekly pivot levels to 6h timeframe
    # Each weekly pivot level applies to the entire week until next weekly bar
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1w, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1w, L4)
    
    # Calculate Donchian(20) on 6h data
    lookback = 20
    # Donchian high: highest high over past 20 bars (including current?)
    # Standard Donchian breakout: break above highest high of past N bars
    # We'll use past 20 bars excluding current to avoid look-ahead
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # For first lookback bars, we don't have enough data
    # But we'll start trading after lookback period anyway
    
    # Calculate average volume for volume confirmation
    vol_lookback = 20
    avg_volume = np.full(n, np.nan)
    for i in range(vol_lookback, n):
        avg_volume[i] = np.mean(volume[i-vol_lookback:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start index: need warmup for Donchian (20) and volume avg (20)
    start_idx = max(lookback, vol_lookback)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume confirmation: current volume > 1.3 * average volume
        volume_confirm = volume[i] > 1.3 * avg_volume[i]
        
        # Determine weekly bias from Camarilla levels
        # Bull bias: price above weekly H3
        # Bear bias: price below weekly L3
        # Neutral: between H3 and L3
        if close[i] > H3_aligned[i]:
            weekly_bias = 'bull'
        elif close[i] < L3_aligned[i]:
            weekly_bias = 'bear'
        else:
            weekly_bias = 'neutral'
        
        if position == 0:
            # Long setup: price breaks above Donchian high AND volume confirm AND bull bias
            long_setup = (close[i] > highest_high[i]) and volume_confirm and (weekly_bias == 'bull')
            
            # Short setup: price breaks below Donchian low AND volume confirm AND bear bias
            short_setup = (close[i] < lowest_low[i]) and volume_confirm and (weekly_bias == 'bear')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_setup:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: price closes below Donchian low OR bear bias OR max holding period (20 bars = ~5 days)
            if (close[i] < lowest_low[i]) or (weekly_bias == 'bear') or (bars_since_entry >= 20):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price closes above Donchian high OR bull bias OR max holding period (20 bars = ~5 days)
            if (close[i] > highest_high[i]) or (weekly_bias == 'bull') or (bars_since_entry >= 20):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_v1"
timeframe = "6h"
leverage = 1.0