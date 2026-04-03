#!/usr/bin/env python3
"""
Experiment #135: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Combining 6h Donchian(20) breakouts with weekly pivot levels provides high-probability entries.
In bull markets: long when price breaks above Donchian(20) high AND above weekly pivot (support/resistance flip).
In bear markets: short when price breaks below Donchian(20) low AND below weekly pivot.
Volume confirmation filters out false breakouts. Uses 6h timeframe for balance of signal quality and trade frequency.
Target: 75-150 total trades over 4 years (19-37/year). Works in both bull/bear by aligning with weekly structure.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot levels ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivots to 6h timeframe (use previous completed week)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # === 6h Indicators ===
    # Donchian Channel (20 periods)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period volume average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(pivot_1w_aligned[i]) or
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation ---
        volume_ok = volume[i] > volume_ma[i]  # Above average volume
        
        # --- Donchian Breakout Signals ---
        donchian_breakout_up = close[i] > donchian_high[i-1]  # Break above previous period's high
        donchian_breakout_down = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # --- Weekly Pivot Alignment ---
        # Price above weekly pivot = bullish bias, below = bearish bias
        price_above_pivot = close[i] > pivot_1w_aligned[i]
        price_below_pivot = close[i] < pivot_1w_aligned[i]
        
        # --- Entry Logic ---
        if not in_position:
            # Long: Donchian breakout up + above weekly pivot + volume confirmation
            if donchian_breakout_up and price_above_pivot and volume_ok:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            # Short: Donchian breakout down + below weekly pivot + volume confirmation
            elif donchian_breakout_down and price_below_pivot and volume_ok:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
        
        # --- Exit Logic ---
        else:
            # Exit conditions: Donchian breakout in opposite direction OR price returns to weekly pivot
            if position_side > 0:  # Long position
                if donchian_breakout_down or close[i] < pivot_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if donchian_breakout_up or close[i] > pivot_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
    
    return signals