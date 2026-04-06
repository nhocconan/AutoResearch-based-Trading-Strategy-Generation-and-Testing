#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Spike
Hypothesis: Combines price channel breakouts with weekly pivot bias and volume confirmation
to capture momentum in trending markets while avoiding chop. Weekly pivot provides
longer-term bias (1d timeframe) to filter breakouts, working in both bull (breakouts
with pivot support) and bear (breakdowns with pivot resistance). Designed for
moderate trade frequency (~15-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1dweeklypivot_vol_v1"
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
    
    # 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly pivot points (using prior week's data)
    # Calculate weekly high/low/close from daily data
    # We'll use 5-day lookback for weekly pivot (approximation)
    weekly_high = np.full(len(high_1d), np.nan)
    weekly_low = np.full(len(low_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    if len(high_1d) >= 5:
        for i in range(5, len(high_1d)):
            weekly_high[i] = np.max(high_1d[i-5:i])
            weekly_low[i] = np.min(low_1d[i-5:i])
            weekly_close[i] = close_1d[i-1]  # Previous day's close
    
    # Weekly pivot point and support/resistance levels
    # Pivot = (H + L + C) / 3
    weekly_pivot = np.full(len(high_1d), np.nan)
    r1 = np.full(len(high_1d), np.nan)
    s1 = np.full(len(high_1d), np.nan)
    r2 = np.full(len(high_1d), np.nan)
    s2 = np.full(len(high_1d), np.nan)
    
    valid = ~(np.isnan(weekly_high) | np.isnan(weekly_low) | np.isnan(weekly_close))
    if np.any(valid):
        weekly_pivot[valid] = (weekly_high[valid] + weekly_low[valid] + weekly_close[valid]) / 3.0
        r1[valid] = 2 * weekly_pivot[valid] - weekly_low[valid]
        s1[valid] = 2 * weekly_pivot[valid] - weekly_high[valid]
        r2[valid] = weekly_pivot[valid] + (weekly_high[valid] - weekly_low[valid])
        s2[valid] = weekly_pivot[valid] - (weekly_high[valid] - weekly_low[valid])
    
    # Trend bias based on weekly pivot:
    # Above R1 = bullish, below S1 = bearish, between = neutral
    weekly_bias = np.full(len(high_1d), 0)  # 0: neutral, 1: bullish, -1: bearish
    bullish = (weekly_close > r1) & ~np.isnan(weekly_close)
    bearish = (weekly_close < s1) & ~np.isnan(weekly_close)
    weekly_bias[bullish] = 1
    weekly_bias[bearish] = -1
    
    # Align to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(weekly_bias_aligned[i]):
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
        volume_filter = volume[i] > vol_ma * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR against weekly bias
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                weekly_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR against weekly bias
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                weekly_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + weekly bias + volume spike
            # Minimum holding period: only allow new entry after 25 bars flat
            if bars_since_entry >= 25:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Long: bullish breakout with bullish weekly bias and volume
                if bull_breakout and weekly_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish breakout with bearish weekly bias and volume
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