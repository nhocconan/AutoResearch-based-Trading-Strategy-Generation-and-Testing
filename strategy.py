#!/usr/bin/env python3
"""
Experiment #2867: 6h Donchian Breakout + Weekly Pivot Direction + Volume Spike
HYPOTHESIS: 6h Donchian(20) breakouts aligned with weekly pivot (from 1w) direction 
capture institutional flow. Weekly pivot acts as structural support/resistance - 
breakouts in direction of weekly trend have higher follow-through. Volume spike 
confirms participation. This combines price structure (Donchian), higher timeframe 
context (weekly pivot), and execution timing (volume spike) to work in both bull 
and bear markets by following the weekly trend. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2867_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for pivot points (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Weekly support/resistance levels
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + range_1w
    s2_1w = pivot_1w - range_1w
    
    # Align to 6h timeframe (shifted by 1 for completed bars only)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit on Donchian opposite breakout (trailing stop via structure)
            if position_side > 0:  # Long
                if price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian HIGH with weekly bias UP
            # Weekly bias UP = price above weekly pivot
            weekly_bias_up = price > pivot_1w  # Use current weekly pivot for bias
            if price > highest_high[i] and weekly_bias_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian LOW with weekly bias DOWN
            # Weekly bias DOWN = price below weekly pivot
            elif price < lowest_low[i] and not weekly_bias_up:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals