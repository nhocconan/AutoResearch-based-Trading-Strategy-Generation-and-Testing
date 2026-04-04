#!/usr/bin/env python3
"""
Experiment #5127: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction from 1d timeframe capture institutional participation. Volume > 1.5x average confirms breakout strength. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag. Weekly pivot provides structural bias that works in both bull (breakouts above R1) and bear (breakdowns below S1) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5127_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Pivot (using prior week's OHLC) ===
    if len(df_1d) >= 5:
        # Calculate weekly OHLC from daily data
        # We'll use the prior completed week's data (shifted by 1 week)
        df_1d_copy = df_1d.copy()
        df_1d_copy['week'] = pd.DatetimeIndex(df_1d_copy.index).isocalendar().week
        df_1d_copy['year'] = pd.DatetimeIndex(df_1d_copy.index).year
        
        # Group by year-week to get weekly OHLC
        weekly = df_1d_copy.groupby(['year', 'week']).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).reset_index()
        
        if len(weekly) >= 2:
            # Use prior week's OHLC for current week's pivot
            prior_week = weekly.iloc[-2]  # Second to last week
            wk_high = prior_week['high']
            wk_low = prior_week['low']
            wk_close = prior_week['close']
            
            # Standard pivot calculation
            pivot = (wk_high + wk_low + wk_close) / 3.0
            r1 = 2 * pivot - wk_low
            s1 = 2 * pivot - wk_high
            r2 = pivot + (wk_high - wk_low)
            s2 = pivot - (wk_high - wk_low)
            r3 = wk_high + 2 * (pivot - wk_low)
            s3 = wk_low - 2 * (wk_high - pivot)
            
            # Align to 6h timeframe: each weekly pivot applies to the entire following week
            # We'll create arrays and align them
            pivot_arr = np.full(len(df_1d), pivot)
            r1_arr = np.full(len(df_1d), r1)
            s1_arr = np.full(len(df_1d), s1)
            r2_arr = np.full(len(df_1d), r2)
            s2_arr = np.full(len(df_1d), s2)
            r3_arr = np.full(len(df_1d), r3)
            s3_arr = np.full(len(df_1d), s3)
            
            # Align to 6h timeframe
            pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_arr)
            r1_6h = align_htf_to_ltf(prices, df_1d, r1_arr)
            s1_6h = align_htf_to_ltf(prices, df_1d, s1_arr)
            r2_6h = align_htf_to_ltf(prices, df_1d, r2_arr)
            s2_6h = align_htf_to_ltf(prices, df_1d, s2_arr)
            r3_6h = align_htf_to_ltf(prices, df_1d, r3_arr)
            s3_6h = align_htf_to_ltf(prices, df_1d, s3_arr)
        else:
            # Not enough weekly data
            pivot_6h = r1_6h = s1_6h = r2_6h = s2_6h = r3_6h = s3_6h = np.full(n, np.nan)
    else:
        pivot_6h = r1_6h = s1_6h = r2_6h = s2_6h = r3_6h = s3_6h = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
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
    
    warmup = max(20, 20)  # Donchian, Volume MA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse on opposite signal ---
        if in_position:
            # Exit conditions: reverse when price crosses opposite pivot level
            if position_side > 0:  # Long
                # Exit long if price breaks below S1 (potential reversal)
                if price <= s1_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit short if price breaks above R1 (potential reversal)
                if price >= r1_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with weekly pivot bias
        # Long: Donchian breakout above + price > R1 (bullish bias)
        # Short: Donchian breakdown below + price < S1 (bearish bias)
        breakout_long = (price >= high_roll[i]) and (price > r1_6h[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < s1_6h[i]) and vol_confirm
        
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