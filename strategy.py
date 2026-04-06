#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Filter
Hypothesis: Donchian breakouts capture momentum, weekly pivot provides directional bias from higher timeframe,
volume confirms breakout strength. Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee decay.
Works in both bull and bear markets by using weekly pivot to filter direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weeklypivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        weekly_pivot = np.full(n, np.nan)
        weekly_r4 = np.full(n, np.nan)
        weekly_s4 = np.full(n, np.nan)
    else:
        weekly_high = df_weekly['high'].values
        weekly_low = df_weekly['low'].values
        weekly_close = df_weekly['close'].values
        
        # Calculate weekly pivot points (standard)
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_range = weekly_high - weekly_low
        weekly_r4 = weekly_pivot + (weekly_range * 3)  # R4 = pivot + 3*range
        weekly_s4 = weekly_pivot - (weekly_range * 3)  # S4 = pivot - 3*range
        
        # Align to 6h timeframe (with shift(1) for completed weekly bars only)
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
        weekly_r4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r4)
        weekly_s4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(weekly_pivot_aligned[i]):
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
            # Look for entries: Donchian breakout + volume + weekly pivot filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Weekly pivot filter: only long above weekly pivot, short below weekly pivot
            # Use R4/S4 for stronger breakout confirmation
            weekly_pivot_val = weekly_pivot_aligned[i]
            weekly_r4_val = weekly_r4_aligned[i]
            weekly_s4_val = weekly_s4_aligned[i]
            
            # Long: price above weekly pivot AND breaking above R4 (strong bullish)
            # Short: price below weekly pivot AND breaking below S4 (strong bearish)
            if bull_breakout and volume_filter and close[i] > weekly_pivot_val and close[i] > weekly_r4_val:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and close[i] < weekly_pivot_val and close[i] < weekly_s4_val:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals