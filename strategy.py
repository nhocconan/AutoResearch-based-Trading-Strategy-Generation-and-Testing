#!/usr/bin/env python3
# 6h_supertrend_4h1d_ema_filter_v1
# Hypothesis: Uses Supertrend (ATR-based) on 4h for trend direction, with 1d EMA200 filter for multi-timeframe alignment.
# Enters long when price closes above Supertrend on 4h and price > 1d EMA200; short when price closes below Supertrend and price < 1d EMA200.
# Exits when price crosses back across Supertrend or 1d EMA200 fails.
# Designed for 15-30 trades/year on 6h to avoid fee drag. Works in bull/bear via trend-following with strong filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_supertrend_4h1d_ema_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4-hour data for Supertrend
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 1-day data for EMA200 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate ATR for Supertrend (10-period)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 9 + tr[i]) / 10  # Wilder's smoothing
    
    # Supertrend parameters
    factor = 3.0
    upperband = (high_4h + low_4h) / 2 + factor * atr
    lowerband = (high_4h + low_4h) / 2 - factor * atr
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_4h)
    uptrend = np.ones_like(close_4h, dtype=bool)
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > upperband[i-1]:
            uptrend[i] = True
        elif close_4h[i] < lowerband[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if not uptrend[i] and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        if uptrend[i]:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    # 1-day EMA200
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h Supertrend and 1d EMA200 to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Ensure EMA200 is ready
    
    for i in range(start_idx, n):
        if np.isnan(supertrend_aligned[i]) or np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend alignment filter
        price_above_ema = close[i] > ema200_1d_aligned[i]
        price_below_ema = close[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Supertrend or below EMA200
            if close[i] < supertrend_aligned[i] or not price_above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Supertrend or above EMA200
            if close[i] > supertrend_aligned[i] or not price_below_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price closes above Supertrend and above EMA200
            if close[i] > supertrend_aligned[i] and price_above_ema:
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below Supertrend and below EMA200
            elif close[i] < supertrend_aligned[i] and price_below_ema:
                position = -1
                signals[i] = -0.25
    
    return signals