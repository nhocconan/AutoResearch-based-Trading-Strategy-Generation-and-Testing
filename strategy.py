#!/usr/bin/env python3
"""
Experiment #094: 1h Strategy with 4h/1d Direction Filters
HYPOTHESIS: Use 4h Donchian breakout + 1d EMA trend for signal direction, 
1h only for precise entry timing with volume confirmation. 
Session filter (08-20 UTC) reduces noise. Discrete sizing 0.20.
Targets 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.
Works in bull/bear by trading breakouts in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_1d_direction_1h_entry_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) - avoid datetime arithmetic in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 4h data for Donchian breakout levels (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    dc_upper_20 = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().shift(1).values
    dc_upper_20_aligned = align_htf_to_ltf(prices, df_4h, dc_upper_20)
    dc_lower_20_aligned = align_htf_to_ltf(prices, df_4h, dc_lower_20)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === 1h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20_aligned[i]) or np.isnan(dc_lower_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- HTF Direction Filters ---
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        # --- Price Channel Breakout (from 4h) ---
        bullish_breakout = close[i] > dc_upper_20_aligned[i]
        bearish_breakout = close[i] < dc_lower_20_aligned[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal or opposite breakout
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold
            if min_hold:
                if position_side > 0:
                    # Exit long: trend turns bearish OR price breaks below 4h lower Donchian
                    if not trend_bullish or close[i] < dc_lower_20_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: trend turns bullish OR price breaks above 4h upper Donchian
                    if not trend_bearish or close[i] > dc_upper_20_aligned[i]:
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
        # Breakout above 4h upper Donchian with bullish 1d EMA trend and volume confirmation
        if bullish_breakout and trend_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below 4h lower Donchian with bearish 1d EMA trend and volume confirmation
        elif bearish_breakout and trend_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals