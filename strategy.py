#!/usr/bin/env python3
"""
6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
Hypothesis: Price breaks through 20-period high/low with volume confirmation
and alignment with weekly pivot trend. Weekly pivot direction acts as trend filter
to avoid counter-trend trades. Works in bull (breakouts above pivot) and bear
(breakdowns below pivot) by only taking breakouts in the direction of weekly trend.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14319_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    
    # Weekly trend: above R1 = uptrend, below S1 = downtrend, between = range
    # We'll use: bullish if close > R1, bearish if close < S1
    weekly_bullish = close_w > r1_w
    weekly_bearish = close_w < s1_w
    
    # Align weekly trend to 6h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_w, weekly_bearish.astype(float))
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_len = 20
    highest_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(donchian_len, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr[i]) or \
           np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: reversal signal (price crosses back below Donchian low) OR stoploss
            if close[i] <= lowest_low[i] or close[i] <= entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: reversal signal (price crosses back above Donchian high) OR stoploss
            if close[i] >= highest_high[i] or close[i] >= entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly trend alignment
            # Long: break above Donchian high + volume + weekly bullish
            long_breakout = close[i] > highest_high[i]
            # Short: break below Donchian low + volume + weekly bearish
            short_breakout = close[i] < lowest_low[i]
            
            if long_breakout and vol_confirm[i] and weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_confirm[i] and weekly_bearish_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals