#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1D Weekly Pivot Direction + Volume Confirmation + ATR Stoploss
Hypothesis: Combines daily trend via weekly pivot (from 1D) with 6H momentum via Donchian breakout.
Weekly pivot provides directional bias from higher timeframe, reducing whipsaw in ranging markets.
Volume confirmation ensures breakout strength. ATR stop limits downside. Designed for 6H timeframe
to target 75-200 total trades over 4 years (~19-50/year) to balance opportunity with fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1dweeklypivot_volume_atr_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        atr[0] = np.nan
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get 1D data once before loop (as required by rules)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from daily data
    # Weekly pivot: (Prior week's High + Low + Close) / 3
    # We'll use prior week's data by shifting the weekly aggregation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high, low, close from daily data
    # Group into weeks (7 days) and aggregate
    weeks_high = []
    weeks_low = []
    weeks_close = []
    
    for i in range(0, len(high_1d), 7):
        week_high = np.max(high_1d[i:i+7]) if len(high_1d[i:i+7]) > 0 else np.nan
        week_low = np.min(low_1d[i:i+7]) if len(low_1d[i:i+7]) > 0 else np.nan
        week_close = close_1d[i+6] if i+6 < len(close_1d) else close_1d[-1]
        weeks_high.append(week_high)
        weeks_low.append(week_low)
        weeks_close.append(week_close)
    
    # Calculate weekly pivot for each week
    weeks_pivot = []
    for wh, wl, wc in zip(weeks_high, weeks_low, weeks_close):
        if not (np.isnan(wh) or np.isnan(wl) or np.isnan(wc)):
            pivot = (wh + wl + wc) / 3.0
        else:
            pivot = np.nan
        weeks_pivot.append(pivot)
    
    # Create daily array of weekly pivot (each day gets the prior week's pivot)
    daily_pivot = np.full(len(high_1d), np.nan)
    for wi in range(len(weeks_pivot)):
        start_idx = wi * 7
        end_idx = min((wi + 1) * 7, len(high_1d))
        if wi > 0:  # Use prior week's pivot
            pivot_val = weeks_pivot[wi-1]
            for di in range(start_idx, end_idx):
                if di < len(daily_pivot):
                    daily_pivot[di] = pivot_val
    
    # Align weekly pivot to 6H timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 14)  # For Donchian and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(pivot_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + pivot direction filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Pivot direction: only long if price above weekly pivot, short if below
            price_above_pivot = close[i] > pivot_aligned[i]
            price_below_pivot = close[i] < pivot_aligned[i]
            
            if bull_breakout and volume_filter and price_above_pivot:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and price_below_pivot:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals