#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Combines 6h Donchian breakouts with weekly pivot direction (from 1w data) to filter for trend-aligned breakouts, plus volume confirmation (>1.5x average) to avoid false breakouts. Weekly pivot provides structural support/resistance from higher timeframe, improving performance in both bull and bear markets by ensuring trades align with weekly bias. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need 100 for weekly pivot + 20 for Donchian
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
    
    # Weekly pivot points (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot: P = (H + L + C)/3
    pivot_1w = np.full(len(close_1w), np.nan)
    r1_1w = np.full(len(close_1w), np.nan)
    s1_1w = np.full(len(close_1w), np.nan)
    r2_1w = np.full(len(close_1w), np.nan)
    s2_1w = np.full(len(close_1w), np.nan)
    
    valid = ~(np.isnan(high_1w) | np.isnan(low_1w) | np.isnan(close_1w))
    if np.any(valid):
        pivot_1w[valid] = (high_1w[valid] + low_1w[valid] + close_1w[valid]) / 3.0
        r1_1w[valid] = 2 * pivot_1w[valid] - low_1w[valid]
        s1_1w[valid] = 2 * pivot_1w[valid] - high_1w[valid]
        r2_1w[valid] = pivot_1w[valid] + (high_1w[valid] - low_1w[valid])
        s2_1w[valid] = pivot_1w[valid] - (high_1w[valid] - low_1w[valid])
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 100  # For weekly pivot calculation
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(pivot_1w_aligned[i]):
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
        
        # Weekly pivot trend filter: price above/below pivot
        above_pivot = close[i] > pivot_1w_aligned[i]
        below_pivot = close[i] < pivot_1w_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below weekly S1 or Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < s1_1w_aligned[i] or
                close[i] < lowest_low or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above weekly R1 or Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > r1_1w_aligned[i] or
                close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + volume + weekly pivot filter
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Long: breakout above resistance AND above weekly pivot
                if bull_breakout and volume_filter and above_pivot:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakout below support AND below weekly pivot
                elif bear_breakout and volume_filter and below_pivot:
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