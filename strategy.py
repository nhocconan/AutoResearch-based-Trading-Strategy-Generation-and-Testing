#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
Hypothesis: 6h Donchian breakouts capture medium-term momentum. Weekly pivot (R1/S1) 
from 1w timeframe filters for institutional bias, reducing false breakouts. Volume 
spike confirms participation. Works in bull/bear via pivot direction filter.
Target: 12-37 trades/year (50-150 over 4 years) on BTC/ETH/SOL.
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
    
    # 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for volatility filter and trailing stop
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly pivot points (MTF: 1w)
    df_1w = get_htf_data(prices, '1w')
    # Typical price for weekly pivot calculation
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    # Weekly pivot: (H+L+C)/3
    pivot_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    # Weekly R1/S1: R1 = 2*P - L, S1 = 2*P - H
    r1_1w = 2 * pivot_1w - df_1w['low']
    s1_1w = 2 * pivot_1w - df_1w['high']
    
    # Align weekly levels to 6h timeframe (with 1-bar delay for completed weekly bar)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w.values)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w.values)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(20, 30) + 1  # Donchian(20) + HTF buffer
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_14[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Donchian breakout conditions
        breakout_long = curr_close > highest_20[i]
        breakout_short = curr_close < lowest_20[i]
        
        # Weekly pivot bias: long above pivot, short below pivot
        long_bias = curr_close > pivot_1w_aligned[i]
        short_bias = curr_close < pivot_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + weekly pivot bias
            long_entry = breakout_long and vol_spike and long_bias
            short_entry = breakout_short and vol_spike and short_bias
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management: ATR trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            exit_level = highest_since_entry - (2.5 * atr_14[i])
            
            if curr_close < exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management: ATR trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            exit_level = lowest_since_entry + (2.5 * atr_14[i])
            
            if curr_close > exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0