#!/usr/bin/env python3
"""
Experiment #147: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot direction from 1d timeframe
capture institutional flow with lower whipsaw. Weekly pivot (calculated from prior week)
provides structural support/resistance levels that work in both bull and bear markets.
Volume confirmation (1.5x average) ensures participation. Targets 12-37 trades/year on
6h timeframe to minimize fee drag while maintaining statistical validity.
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
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week's OHLC
    # Using prior week's high, low, close to compute standard pivot levels
    week_high = df_1d['high'].values
    week_low = df_1d['low'].values
    week_close = df_1d['close'].values
    
    # Standard pivot point calculation: P = (H + L + C) / 3
    pivot_point = (week_high + week_low + week_close) / 3.0
    
    # Support and resistance levels
    r1 = 2 * pivot_point - week_low
    s1 = 2 * pivot_point - week_high
    r2 = pivot_point + (week_high - week_low)
    s2 = pivot_point - (week_high - week_low)
    r3 = week_high + 2 * (pivot_point - week_low)
    s3 = week_low - 2 * (week_high - pivot_point)
    
    # Align HTF pivot levels to LTF (6h) with shift(1) for completed bars only
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 6h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Direction ---
        # Bullish bias: price above weekly pivot and R3
        pivot_bullish = close[i] > pivot_aligned[i] and close[i] > r3_aligned[i]
        # Bearish bias: price below weekly pivot and S3
        pivot_bearish = close[i] < pivot_aligned[i] and close[i] < s3_aligned[i]
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~18h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR breaks below pivot/S1
                    if close[i] <= dc_lower_20[i] or close[i] < s1_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR breaks above pivot/R1
                    if close[i] >= dc_upper_20[i] or close[i] > r1_aligned[i]:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # Breakout above upper Donchian with bullish weekly pivot bias and volume confirmation
        if bullish_breakout and pivot_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish weekly pivot bias and volume confirmation
        elif bearish_breakout and pivot_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals