#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_Volume
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (bullish/bearish) and volume confirmation (>2.0x 20-bar avg) captures institutional breakouts with controlled frequency. Weekly pivot direction from 1w HTF ensures alignment with higher-timeframe structure, reducing false breakouts. Discrete sizing (0.25) minimizes fee churn. Works in bull markets via long breakouts and bear markets via short breakouts. Uses 1d HTF for Donchian calculation and 1w for pivot direction to avoid look-ahead.
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
    
    # Get 1d data for HTF Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 1d for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high of last 20 daily bars
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 daily bars
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate weekly pivot points from previous completed 1w bar
    # Standard pivot: P = (H + L + C) / 3
    # Support 1: S1 = (2*P) - H
    # Resistance 1: R1 = (2*P) - L
    # We'll use the bias: if current weekly close > weekly pivot -> bullish bias
    # else bearish bias
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Use previous completed 1w bar to avoid look-ahead
    prev_high_1w = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low_1w = np.concatenate([[np.nan], low_1w[:-1]])
    prev_close_1w = np.concatenate([[np.nan], close_1w[:-1]])
    
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    # Bullish bias: weekly close > weekly pivot
    bullish_bias = prev_close_1w > pivot_1w
    bearish_bias = prev_close_1w < pivot_1w
    
    # Align bias to 6h timeframe (as boolean arrays)
    bullish_bias_aligned = align_htf_to_ltf(prices, df_1w, bullish_bias.astype(float))
    bearish_bias_aligned = align_htf_to_ltf(prices, df_1w, bearish_bias.astype(float))
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(20, 20)  # Donchian20, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(bullish_bias_aligned[i]) or 
            np.isnan(bearish_bias_aligned[i]) or 
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
        upper_val = upper_20_aligned[i]
        lower_val = lower_20_aligned[i]
        bullish_val = bullish_bias_aligned[i]
        bearish_val = bearish_bias_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Donchian breakout with weekly pivot direction and volume
            # Long: price breaks above upper Donchian with bullish weekly bias and volume spike
            long_signal = (high_val > upper_val) and (bullish_val > 0.5) and volume_spike
            # Short: price breaks below lower Donchian with bearish weekly bias and volume spike
            short_signal = (low_val < lower_val) and (bearish_val > 0.5) and volume_spike
            
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
            # 1. Opposite breakout: price breaks below lower Donchian (exit long)
            if close_val < lower_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Weekly bias flips to bearish (exit long)
            elif bearish_val > 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above upper Donchian (exit short)
            if close_val > upper_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Weekly bias flips to bullish (exit short)
            elif bullish_val > 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0