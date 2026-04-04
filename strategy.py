#!/usr/bin/env python3
"""
Experiment #4739: 6h Donchian(20) Breakout + 12h Volume Spike + HTF Weekly Pivot Direction
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (from 1d HTF) with volume confirmation (>2x average) capture strong momentum moves. Uses ATR(14) trailing stop (2.5x) to limit downside. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakdowns against trend) by using weekly pivot as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4739_6h_donchian20_12h_vol_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for volume MA and 1d for weekly pivot
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # === 12h Indicators: Volume MA(20) for confirmation ===
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    else:
        vol_ma_12h_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Weekly Pivot (using prior week's H/L/C) ===
    if len(df_1d) >= 5:
        # Calculate weekly pivot from prior completed week
        # We need H, L, C from the week before the current bar's week
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Rolling window of 5 days (1 week) to get prior week's HLC
        # We'll use shift(5) to ensure we only use completed weeks
        if len(high_1d) >= 10:  # Need at least 2 weeks
            # Prior week's high/low/close (5 days ago to 1 day ago)
            week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(5)
            week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(5)
            week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(5)
            
            # Classic pivot point: (H + L + C) / 3
            pivot = (week_high + week_low + week_close) / 3.0
            # Support 1: (2 * P) - H
            s1 = (2 * pivot) - week_high
            # Resistance 1: (2 * P) - L
            r1 = (2 * pivot) - week_low
            
            # Align to 6h timeframe
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
            r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
            s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
            
            # Determine bias: price > pivot = bullish bias, price < pivot = bearish bias
            bullish_bias = pivot_aligned > 0  # Will be refined in loop
            bearish_bias = pivot_aligned > 0  # Will be refined in loop
        else:
            pivot_aligned = np.full(n, np.nan)
            r1_aligned = np.full(n, np.nan)
            s1_aligned = np.full(n, np.nan)
    else:
        pivot_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = vol_ma_12h_aligned[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # --- Volume confirmation (>2.0x) ---
        vol_confirm = volume[i] > (2.0 * vol_ma) if vol_ma > 0 else False
        
        # --- Pivot bias determination ---
        bullish_bias = price > pivot_val
        bearish_bias = price < pivot_val
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions with pivot bias alignment
        breakout_long = (price >= high_roll[i]) and bullish_bias and vol_confirm
        breakout_short = (price <= low_roll[i]) and bearish_bias and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals