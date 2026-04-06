#!/usr/bin/env python3
"""
6H WEEKLY PIVOT POINT BREAKOUT + VOLUME CONFIRMATION
Hypothesis: Weekly pivot points act as strong support/resistance. Breakouts above R1 or below S1 with volume confirmation capture momentum. Works in bull (upside breakouts) and bear (downside breakouts). Uses weekly data for pivot calculation to avoid noise. Target: 80-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v1"
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
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    pivot = np.full(len(close_w), np.nan)
    r1 = np.full(len(close_w), np.nan)
    s1 = np.full(len(close_w), np.nan)
    for i in range(len(close_w)):
        if not (np.isnan(high_w[i]) or np.isnan(low_w[i]) or np.isnan(close_w[i])):
            pivot[i] = (high_w[i] + low_w[i] + close_w[i]) / 3.0
            r1[i] = 2 * pivot[i] - low_w[i]
            s1[i] = 2 * pivot[i] - high_w[i]
    
    # Align to 6h timeframe (shifted by 1 week for look-ahead protection)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below weekly pivot or stoploss hit
            if (close[i] < pivot_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above weekly pivot or stoploss hit
            if (close[i] > pivot_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price breaks above R1 with volume
            if (close[i] > r1_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below S1 with volume
            elif (close[i] < s1_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals