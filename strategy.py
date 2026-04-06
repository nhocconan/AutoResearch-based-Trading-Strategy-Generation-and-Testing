#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum aligned with weekly pivot bias (from daily HTF), volume confirms breakout strength. Weekly pivot provides structural bias that works in both bull/bear markets. Target 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1dweeklypivot_vol_v2"
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
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Load 1d data once before loop for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from daily data (using prior week's data)
    # We'll calculate pivot for each day based on prior 5 trading days (1 week)
    n_1d = len(close_1d)
    weekly_pivot = np.full(n_1d, np.nan)
    weekly_r1 = np.full(n_1d, np.nan)
    weekly_s1 = np.full(n_1d, np.nan)
    weekly_r2 = np.full(n_1d, np.nan)
    weekly_s2 = np.full(n_1d, np.nan)
    
    # Need at least 5 days for weekly calculation
    if n_1d >= 5:
        for i in range(5, n_1d):
            # Use prior 5 days (not including current) to calculate weekly pivot
            lookback_high = np.max(high_1d[i-5:i])
            lookback_low = np.min(low_1d[i-5:i])
            lookback_close = close_1d[i-1]  # Previous day's close
            
            pivot = (lookback_high + lookback_low + lookback_close) / 3.0
            weekly_pivot[i] = pivot
            weekly_r1[i] = 2 * pivot - lookback_low
            weekly_s1[i] = 2 * pivot - lookback_high
            weekly_r2[i] = pivot + (lookback_high - lookback_low)
            weekly_s2[i] = pivot - (lookback_high - lookback_low)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(weekly_pivot_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
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
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + volume + weekly pivot filter
            # Minimum holding period: only allow new entry after 8 bars flat
            if bars_since_entry >= 8:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Weekly pivot filter: 
                # Long only if price above weekly pivot (bullish bias)
                # Short only if price below weekly pivot (bearish bias)
                pivot_bias_long = close[i] > weekly_pivot_aligned[i]
                pivot_bias_short = close[i] < weekly_pivot_aligned[i]
                
                if bull_breakout and volume_filter and pivot_bias_long:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter and pivot_bias_short:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals