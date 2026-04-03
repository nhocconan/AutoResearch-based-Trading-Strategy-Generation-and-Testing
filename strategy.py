#!/usr/bin/env python3
"""
Experiment #215: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike + ATR Stoploss

HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot direction capture medium-term momentum with institutional participation. Weekly pivot provides major S/R levels from the prior week, filtering false breakouts. Volume spike confirms breakout validity. This combination works in both bull and bear markets by following established trends while filtering false breakouts. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to balance opportunity with fee drag.
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
    
    # === HTF: 1w data for weekly pivot points (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard formula)
    if len(df_1w) >= 1:
        # Pivot Point = (High + Low + Close) / 3
        pp = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
        # R1 = (2 * Pivot) - Low
        r1 = (2 * pp) - df_1w['low']
        # S1 = (2 * Pivot) - High
        s1 = (2 * pp) - df_1w['high']
        # R2 = Pivot + (High - Low)
        r2 = pp + (df_1w['high'] - df_1w['low'])
        # S2 = Pivot - (High - Low)
        s2 = pp - (df_1w['high'] - df_1w['low'])
        # R3 = High + 2*(Pivot - Low)
        r3 = df_1w['high'] + 2 * (pp - df_1w['low'])
        # S3 = Low - 2*(High - Pivot)
        s3 = df_1w['low'] - 2 * (df_1w['high'] - pp)
        
        # Use R3/S3 as major resistance/support for breakout confirmation
        # Align to 6h timeframe
        r3_aligned = align_htf_to_ltf(prices, df_1w, r3.values)
        s3_aligned = align_htf_to_ltf(prices, df_1w, s3.values)
    else:
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
    
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
            np.isnan(vol_ma_20[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Weekly Pivot Filter ---
        # Only take longs above weekly R3 (strong resistance turned support)
        # Only take shorts below weekly S3 (strong support turned resistance)
        bullish_pivot = close[i] > r3_aligned[i]
        bearish_pivot = close[i] < s3_aligned[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2x volume spike
        
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
                    # Exit long: price touches lower Donchian OR breaks below weekly S3
                    if close[i] <= dc_lower_20[i] or close[i] < s3_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR breaks above weekly R3
                    if close[i] >= dc_upper_20[i] or close[i] > r3_aligned[i]:
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
        # Breakout above upper Donchian with price above weekly R3 and volume confirmation
        if bullish_breakout and bullish_pivot and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with price below weekly S3 and volume confirmation
        elif bearish_breakout and bearish_pivot and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>