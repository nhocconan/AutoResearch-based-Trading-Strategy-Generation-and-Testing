#!/usr/bin/env python3
"""
Experiment #4335: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts aligned with weekly pivot bias (price above/below weekly pivot) 
and confirmed by volume spikes (>1.5x average) capture institutional breakout moves in both bull 
and bear markets. Weekly pivot provides structural bias, Donchian captures breakouts, volume 
confirms authenticity. Targets 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4335_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1w data for weekly pivot ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Calculate weekly pivot points (standard formula)
        weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
        weekly_r1 = 2 * weekly_pivot - df_1w['low']
        weekly_s1 = 2 * weekly_pivot - df_1w['high']
        weekly_r2 = weekly_pivot + (df_1w['high'] - df_1w['low'])
        weekly_s2 = weekly_pivot - (df_1w['high'] - df_1w['low'])
        
        # Align to 6h timeframe (shifted by 1 for completed weekly bars only)
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1.values)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1.values)
        weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2.values)
        weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2.values)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
        weekly_r2_aligned = np.full(n, np.nan)
        weekly_s2_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
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
    
    warmup = max(lookback, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
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
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Donchian breakout conditions
            breakout_up = price > highest_high[i]  # Price breaks above Donchian upper
            breakout_down = price < lowest_low[i]  # Price breaks below Donchian lower
            
            # Weekly pivot bias
            price_above_pivot = price > weekly_pivot_aligned[i]
            price_below_pivot = price < weekly_pivot_aligned[i]
            
            # Long conditions: Breakout above Donchian upper + price above weekly pivot
            long_entry = breakout_up and price_above_pivot
            
            # Short conditions: Breakout below Donchian lower + price below weekly pivot
            short_entry = breakout_down and price_below_pivot
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals