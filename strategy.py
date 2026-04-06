#!/usr/bin/env python3
"""
6H WEEKLY PIVOT BREAKOUT + VOLUME CONFIRMATION
Hypothesis: Weekly pivot levels act as strong support/resistance. Breakouts above weekly R1 or below weekly S1 with volume confirmation capture institutional flow. Works in bull (breakouts above R1) and bear (breakdowns below S1). Uses 12h trend filter to avoid whipsaws. Target: 50-150 total trades over 4 years.
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
    pivot_w = (high_w + low_w + close_w) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    
    # Align weekly levels to 6h timeframe (shifted by 1 for completed weekly bar)
    pivot_w_aligned = align_htf_to_ltf(prices, df_weekly, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_weekly, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_weekly, s1_w)
    
    # Get 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # EMA(34) on 12h for trend
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_34_12h = ema(close_12h, 34)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume filter: current volume > 1.8x average over last 24 periods
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 34, 24)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below weekly pivot or stoploss hit
            if (close[i] < pivot_w_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above weekly pivot or stoploss hit
            if (close[i] > pivot_w_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price breaks above weekly R1, above 12h EMA34, with volume
            if (close[i] > r1_w_aligned[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below weekly S1, below 12h EMA34, with volume
            elif (close[i] < s1_w_aligned[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals