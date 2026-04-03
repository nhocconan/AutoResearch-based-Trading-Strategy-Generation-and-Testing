#!/usr/bin/env python3
"""
Experiment #1891: 6h Donchian(20) Breakout + 1d Weekly Pivot + Volume Confirmation
HYPOTHESIS: Donchian breakouts capture strong momentum. 1d weekly pivot (from prior week) provides institutional support/resistance levels. Enter long when price breaks above Donchian upper band AND above weekly pivot R1; short when breaks below lower band AND below weekly pivot S1. Volume confirmation (>1.5x average) filters weak breakouts. Works in bull/bear by following price action relative to pivot levels. Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing of 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1891_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from prior week (using 1d data)
    # For each 1d bar, use prior week's high/low/close (shifted by 5 days approx)
    # We'll use prior 5-day aggregate: highest high, lowest low, last close of prior 5 days
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot calculation (standard formula)
    pivot = (high_5d + low_5d + close_5d) / 3.0
    r1 = 2 * pivot - low_5d
    s1 = 2 * pivot - high_5d
    r2 = pivot + (high_5d - low_5d)
    s2 = pivot - (high_5d - low_5d)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
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
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20) and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or adverse move ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below Donchian lower band (20)
                if price < lowest_low[i]:
                    exit_signal = True
                # Exit if price falls below weekly pivot S1
                elif price < s1_aligned[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above Donchian upper band (20)
                if price > highest_high[i]:
                    exit_signal = True
                # Exit if price rises above weekly pivot R1
                elif price > r1_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long: price breaks above Donchian upper band AND above weekly pivot R1
        if (price > highest_high[i] and price > r1_aligned[i] and vol_ratio[i] > 1.5):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: price breaks below Donchian lower band AND below weekly pivot S1
        elif (price < lowest_low[i] and price < s1_aligned[i] and vol_ratio[i] > 1.5):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals