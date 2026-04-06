#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: Weekly pivots capture longer-term market structure; price above/below weekly pivot
indicates bull/bear bias. Breakouts from 6h Donchian channels aligned with weekly bias
and volume confirmation capture momentum while avoiding counter-trend traps.
Works in bull (long bias breakouts) and bear (short bias breakdowns). Target: 75-150 total trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_vol_v2"
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
    
    # 20-period ATR for stops and filters
    atr = np.full(n, np.nan)
    if n >= 20:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[20] = np.mean(tr[:20])
            for i in range(21, n):
                atr[i] = (atr[i-1] * 19 + tr[i-1]) / 20
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Weekly pivot point and key levels
    pivot_w = np.full(len(high_w), np.nan)
    r1_w = np.full(len(high_w), np.nan)
    s1_w = np.full(len(high_w), np.nan)
    r2_w = np.full(len(high_w), np.nan)
    s2_w = np.full(len(high_w), np.nan)
    
    for i in range(len(close_w)):
        if i >= 0:  # Need at least one bar
            pivot_w[i] = (high_w[i] + low_w[i] + close_w[i]) / 3.0
            r1_w[i] = 2 * pivot_w[i] - low_w[i]
            s1_w[i] = 2 * pivot_w[i] - high_w[i]
            r2_w[i] = pivot_w[i] + (high_w[i] - low_w[i])
            s2_w[i] = pivot_w[i] - (high_w[i] - low_w[i])
    
    # Bias: 1 if close > pivot (bullish bias), -1 if close < pivot (bearish bias)
    bias_w = np.where(close_w > pivot_w, 1, -1)
    bias_w_aligned = align_htf_to_ltf(prices, df_weekly, bias_w)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
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
        if np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(bias_w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR bias turns bearish
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] <= donch_low[i] or
                bias_w_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR bias turns bullish
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] >= donch_high[i] or
                bias_w_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high with bullish bias and volume
            if (close[i] > donch_high[i] and
                bias_w_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low with bearish bias and volume
            elif (close[i] < donch_low[i] and
                  bias_w_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals