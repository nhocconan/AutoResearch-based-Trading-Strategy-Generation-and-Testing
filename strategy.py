#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Filter
Hypothesis: Donchian breakouts capture momentum aligned with weekly pivot bias, volume confirms breakout strength.
Weekly pivots provide institutional support/resistance levels that work in both bull and bear markets.
Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee decay.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weeklypivot_vol_v1"
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
    
    # Load weekly data once before loop
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_pivot = np.full(len(weekly_close), np.nan)
    weekly_r1 = np.full(len(weekly_close), np.nan)
    weekly_s1 = np.full(len(weekly_close), np.nan)
    weekly_r2 = np.full(len(weekly_close), np.nan)
    weekly_s2 = np.full(len(weekly_close), np.nan)
    weekly_r3 = np.full(len(weekly_close), np.nan)
    weekly_s3 = np.full(len(weekly_close), np.nan)
    
    valid = ~(np.isnan(weekly_high) | np.isnan(weekly_low) | np.isnan(weekly_close))
    if np.any(valid):
        weekly_pivot[valid] = (weekly_high[valid] + weekly_low[valid] + weekly_close[valid]) / 3.0
        weekly_r1[valid] = 2 * weekly_pivot[valid] - weekly_low[valid]
        weekly_s1[valid] = 2 * weekly_pivot[valid] - weekly_high[valid]
        weekly_r2[valid] = weekly_pivot[valid] + (weekly_high[valid] - weekly_low[valid])
        weekly_s2[valid] = weekly_pivot[valid] - (weekly_high[valid] - weekly_low[valid])
        weekly_r3[valid] = weekly_high[valid] + 2 * (weekly_pivot[valid] - weekly_low[valid])
        weekly_s3[valid] = weekly_low[valid] - 2 * (weekly_high[valid] - weekly_pivot[valid])
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian
    
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
            # Look for entries: Donchian breakout + volume + weekly pivot bias
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Weekly pivot bias: long above pivot, short below pivot
            pivot_bias_long = close[i] > pivot_aligned[i]
            pivot_bias_short = close[i] < pivot_aligned[i]
            
            if bull_breakout and volume_filter and pivot_bias_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and pivot_bias_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals