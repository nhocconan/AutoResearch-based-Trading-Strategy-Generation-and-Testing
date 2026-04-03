#!/usr/bin/env python3
"""
Experiment #299: 6h Donchian(20) breakout + 12h weekly pivot direction + volume confirmation
HYPOTHESIS: Combining Donchian breakouts with weekly pivot bias and volume confirmation captures institutional flow. In bull markets, price breaks above weekly R1 with volume continues up. In bear markets, price breaks below weekly S1 with volume continues down. Weekly pivot provides structural context, Donchian ensures breakout validity, volume confirms conviction. Discrete sizing (0.25) minimizes fee drag. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_299_6h_donchian20_12h_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for weekly pivot (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate weekly pivot points from prior week (5x 12h bars = ~5 days)
    week_high = df_12h['high'].rolling(window=5, min_periods=5).max().shift(1)
    week_low = df_12h['low'].rolling(window=5, min_periods=5).min().shift(1)
    week_close = df_12h['close'].rolling(window=5, min_periods=5).last().shift(1)
    
    pivot = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    r2 = pivot + (week_high - week_low)
    s2 = pivot - (week_high - week_low)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2.values)
    
    # === 6h Indicators: Donchian(20) channels ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Weekly Pivot Bias ---
        # Long bias: above weekly R1 (bullish structure)
        # Short bias: below weekly S1 (bearish structure)
        long_bias = price > r1_aligned[i]
        short_bias = price < s1_aligned[i]
        
        # --- Exit Logic: Mean reversion to weekly pivot or opposite Donchian ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2*ATR approximation using Donchian width
                donch_width = highest_high[i] - lowest_low[i]
                stop_level = entry_price - 1.5 * donch_width
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit: return to weekly pivot
                if price <= pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Reverse signal: break below Donchian low
                if price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2*ATR approximation
                donch_width = highest_high[i] - lowest_low[i]
                stop_level = entry_price + 1.5 * donch_width
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit: return to weekly pivot
                if price >= pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Reverse signal: break above Donchian high
                if price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up AND long bias from weekly pivot
            if breakout_up and long_bias:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down AND short bias from weekly pivot
            elif breakout_down and short_bias:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals