#!/usr/bin/env python3
"""
Experiment #1879: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Donchian breakouts capture strong momentum moves. Weekly pivot (from 1w HTF) provides structural bias: price above weekly pivot = bullish bias (long breakouts only), below = bearish bias (short breakouts only). Volume confirmation (>1.5x average) filters false breakouts. Works in both bull/bear markets by aligning with higher timeframe structure. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1879_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (using prior week's data)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # We'll use prior week's values to avoid look-ahead
    hw_1w = np.roll(high_1w, 1)  # prior week high
    lw_1w = np.roll(low_1w, 1)   # prior week low
    cw_1w = np.roll(close_1w, 1) # prior week close
    hw_1w[0] = high_1w[0]        # first bar uses current week (no prior)
    lw_1w[0] = low_1w[0]
    cw_1w[0] = close_1w[0]
    
    weekly_pivot = (hw_1w + lw_1w + cw_1w) / 3.0
    weekly_r1 = 2 * weekly_pivot - lw_1w  # Resistance 1
    weekly_s1 = 2 * weekly_pivot - hw_1w  # Support 1
    
    # Align weekly pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # === 6h Indicators: Donchian Channel (20) ===
    # Upper = highest high over past 20 periods
    # Lower = lowest low over past 20 periods
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # for Donchian(20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Opposite Donchian breakout or time-based ---
        if in_position:
            bars_since_entry += 1
            
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below Donchian lower (contrarian exit)
                if price < donchian_lower[i]:
                    exit_signal = True
                # Optional: exit if price reaches weekly R1 (take profit)
                elif price >= r1_aligned[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above Donchian upper (contrarian exit)
                if price > donchian_upper[i]:
                    exit_signal = True
                # Optional: exit if price reaches weekly S1 (take profit)
                elif price <= s1_aligned[i]:
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
        # Determine bias from weekly pivot: above pivot = bullish, below = bearish
        bullish_bias = price > pivot_aligned[i]
        bearish_bias = price < pivot_aligned[i]
        
        # Volume confirmation: require significant volume (> 1.5x average)
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Long entry: price breaks above Donchian upper AND bullish bias from weekly pivot
            if price > donchian_upper[i] and bullish_bias:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND bearish bias from weekly pivot
            elif price < donchian_lower[i] and bearish_bias:
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