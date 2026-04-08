#!/usr/bin/env python3
"""
6h Donchian(20) breakout with weekly pivot direction and volume confirmation
Hypothesis: Donchian breakouts capture institutional momentum, filtered by weekly pivot direction (from 1d data) for trend bias and volume for conviction. Works in bull (buy breakouts above weekly pivot) and bear (sell breakdowns below weekly pivot). Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weeklypivot_vol_v1"
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
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from daily data (using prior week)
    # Weekly pivot = (Prior week high + prior week low + prior week close) / 3
    weekly_pivot = np.full(len(high_1d), np.nan)
    weekly_high = np.full(len(high_1d), np.nan)
    weekly_low = np.full(len(high_1d), np.nan)
    weekly_close = np.full(len(high_1d), np.nan)
    
    # Need at least 5 days for a week
    for i in range(5, len(high_1d)):
        # Prior week: i-5 to i-1
        weekly_high[i] = np.max(high_1d[i-5:i])
        weekly_low[i] = np.min(low_1d[i-5:i])
        weekly_close[i] = close_1d[i-1]  # Previous day's close as weekly close
        weekly_pivot[i] = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
    
    # Weekly bias: above pivot = bullish, below = bearish
    weekly_bias = np.where(close_1d > weekly_pivot, 1, -1)
    
    # Align weekly bias to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    
    # Donchian channels (20-period) from 1d data
    upper_1d = np.full_like(high_1d, np.nan)
    lower_1d = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        upper_1d[i] = np.max(high_1d[i-20:i])
        lower_1d[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # Need enough data for Donchian and weekly pivot
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(weekly_bias_aligned[i]) or 
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against weekly bias
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower_aligned[i] or
                weekly_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against weekly bias
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper_aligned[i] or
                weekly_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Breakout entries: upper/lower with weekly bias
                bull_breakout = close[i] > upper_aligned[i]
                bear_breakout = close[i] < lower_aligned[i]
                
                # Long: breakout above upper with bullish weekly bias + volume
                if bull_breakout and weekly_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish weekly bias + volume
                elif bear_breakout and weekly_bias_aligned[i] == -1 and volume_filter:
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