#!/usr/bin/env python3
"""
Experiment #639: 6h Donchian(20) breakout + 12h Camarilla pivot (R3/S3 fade, R4/S4 breakout) + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Camarilla pivot levels (R3/S3 for mean reversion fade, R4/S4 for breakout continuation) capture institutional order flow. Volume confirmation filters low-participation moves. Designed to work in bull markets (breakout continuation at R4/S4) and bear markets (fade at R3/S3 during rallies/drops). Target: 75-200 total trades over 4 years via tight entry conditions requiring confluence of price, pivot levels, and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_639_6h_donchian20_12h_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot levels ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r3_12h = pivot_12h + range_12h * 1.1 / 4.0
    s3_12h = pivot_12h - range_12h * 1.1 / 4.0
    r4_12h = pivot_12h + range_12h * 1.1 / 2.0
    s4_12h = pivot_12h - range_12h * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian and ATR calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(pivot_12h_aligned[i]) or
            np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Camarilla Pivot Logic ---
        # Fade at R3/S3 (mean reversion): price touches R3/S3 and reverses
        # Breakout continuation at R4/S4: price breaks R4/S4 with momentum
        near_r3 = abs(price - r3_12h_aligned[i]) < (r4_12h_aligned[i] - r3_12h_aligned[i]) * 0.05
        near_s3 = abs(price - s3_12h_aligned[i]) < (s3_12h_aligned[i] - s4_12h_aligned[i]) * 0.05
        breakout_r4 = price > r4_12h_aligned[i]
        breakdown_s4 = price < s4_12h_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 10 bars (~60h on 6h) to avoid overtrading
            if bars_since_entry > 10:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long scenarios:
            # 1. Fade at S3: price near S3 and breaking above (mean reversion long)
            # 2. Breakout continuation at R4: price breaks above R4 (momentum long)
            if (near_s3 and breakout_up) or (breakout_r4 and breakout_up):
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short scenarios:
            # 1. Fade at R3: price near R3 and breaking below (mean reversion short)
            # 2. Breakdown continuation at S4: price breaks below S4 (momentum short)
            elif (near_r3 and breakout_down) or (breakdown_s4 and breakout_down):
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