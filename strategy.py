#!/usr/bin/env python3
"""
Experiment #3027: 6h Donchian Breakout + 1d Weekly Pivot Direction + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 6h capture medium-term swings. Weekly pivot
direction from 1d data provides institutional bias: only take longs when price >
weekly pivot point, shorts when price < weekly pivot. Volume spike (>2.0x 20-period
average) confirms breakout strength. This filters false breakouts while capturing
strong trends in both bull and bear markets. 6h timeframe balances trade frequency
and fee drag. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3027_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week (using last 5 daily bars)
    def calculate_weekly_pivot(high_arr, low_arr, close_arr):
        """Calculate weekly pivot from last 5 daily bars (prior week)"""
        if len(high_arr) < 5:
            return np.full_like(high_arr, np.nan), np.full_like(high_arr, np.nan), np.full_like(high_arr, np.nan)
        
        # Use last 5 days (prior week) for weekly pivot
        week_high = np.max(high_arr[-5:])
        week_low = np.min(low_arr[-5:])
        week_close = close_arr[-1]  # Most recent daily close
        
        pivot = (week_high + week_low + week_close) / 3.0
        r1 = 2 * pivot - week_low
        s1 = 2 * pivot - week_high
        r2 = pivot + (week_high - week_low)
        s2 = pivot - (week_high - week_low)
        r3 = week_high + 2 * (pivot - week_low)
        s3 = week_low - 2 * (week_high - pivot)
        
        # Return arrays aligned to length (same value for all bars)
        pivot_arr = np.full_like(high_arr, pivot)
        r1_arr = np.full_like(high_arr, r1)
        s1_arr = np.full_like(high_arr, s1)
        r2_arr = np.full_like(high_arr, r2)
        s2_arr = np.full_like(high_arr, s2)
        r3_arr = np.full_like(high_arr, r3)
        s3_arr = np.full_like(high_arr, s3)
        
        return pivot_arr, r1_arr, s1_arr, r2_arr, s2_arr, r3_arr, s3_arr
    
    # Calculate weekly pivot for each point (using expanding window of prior week)
    pivot_vals = np.full(n, np.nan)
    r1_vals = np.full(n, np.nan)
    s1_vals = np.full(n, np.nan)
    r2_vals = np.full(n, np.nan)
    s2_vals = np.full(n, np.nan)
    r3_vals = np.full(n, np.nan)
    s3_vals = np.full(n, np.nan)
    
    # Calculate weekly pivot for each bar using prior 5 days
    for i in range(5, len(high_1d)):
        week_high = np.max(high_1d[i-5:i])
        week_low = np.min(low_1d[i-5:i])
        week_close = close_1d[i-1]  # Prior day close
        
        pivot = (week_high + week_low + week_close) / 3.0
        r1 = 2 * pivot - week_low
        s1 = 2 * pivot - week_high
        r2 = pivot + (week_high - week_low)
        s2 = pivot - (week_high - week_low)
        r3 = week_high + 2 * (pivot - week_low)
        s3 = week_low - 2 * (week_high - pivot)
        
        pivot_vals[i] = pivot
        r1_vals[i] = r1
        s1_vals[i] = s1
        r2_vals[i] = r2
        s2_vals[i] = s2
        r3_vals[i] = r3
        s3_vals[i] = s3
    
    # Align HTF weekly pivot to LTF (6h)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_vals)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_vals)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_vals)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_vals)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_vals)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_vals)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                atr_estimate = (high[i] - low[i]) * 0.5
                if price < highest_since_entry - 2.5 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price <= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                atr_estimate = (high[i] - low[i]) * 0.5
                if price > lowest_since_entry + 2.5 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Get weekly pivot levels
            pivot = pivot_1d_aligned[i]
            r1 = r1_1d_aligned[i]
            s1 = s1_1d_aligned[i]
            
            # Long entry: price breaks above Donchian high with price > weekly pivot
            if price > highest_high[i] and price > pivot:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with price < weekly pivot
            elif price < lowest_low[i] and price < pivot:
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