#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + ATR Regime + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum. ATR-based regime filter
distinguishes trending (ATR rising) from ranging (ATR falling) markets to avoid false breakouts.
Volume spike confirms institutional participation. Works in both bull and bear via symmetric logic.
Target 20-30 trades/year on 4h to stay within fee drag limits.
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
    
    # Get 1d data for ATR regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for regime filter
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            abs(high_1d - close_1d.shift(1)),
            abs(low_1d - close_1d.shift(1))
        )
    )
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 4h ATR(14) for stop management
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Donchian, ATR, volume MA
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        atr_1d_val = atr_14_1d_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        
        # Regime filter: ATR rising = trending market (good for breakouts)
        # Compare current 1d ATR to its 5-period ago value
        atr_rising = False
        if i >= 5:  # Need history to check ATR trend
            atr_1d_prev = atr_14_1d_aligned[i-5]
            atr_rising = atr_1d_val > atr_1d_prev * 1.1  # 10% increase = rising volatility
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above upper Donchian channel with volume confirmation in rising ATR regime
            long_breakout = (curr_close > upper_channel) and volume_confirm and atr_rising
            # Short: price breaks below lower Donchian channel with volume confirmation in rising ATR regime
            short_breakout = (curr_close < lower_channel) and volume_confirm and atr_rising
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price closes below lower channel OR 2.5*ATR trailing stop
            if curr_close < lower_channel or curr_close < (highest_since_entry - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above upper channel OR 2.5*ATR trailing stop
            if curr_close > upper_channel or curr_close > (lowest_since_entry + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_ATR_Regime_VolumeSpike"
timeframe = "4h"
leverage = 1.0