#!/usr/bin/env python3
# 6h_12h_1d_camarilla_breakout_v1
# Hypothesis: 6-hour breakouts above/below daily Camarilla pivot levels (H4/L4) with 12h trend filter.
# Long when price breaks above H4 with 12h close > open (bullish candle).
# Short when price breaks below L4 with 12h close < open (bearish candle).
# Exit when price returns to the daily pivot point (PP).
# Uses tight entry conditions to limit trades and avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # H4 and L4 levels (stronger breakout levels)
    h4_1d = close_1d + (range_1d * 1.1 / 2)
    l4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Align 1d levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h bullish/bearish candle: close > open = bullish, close < open = bearish
    open_12h = df_12h['open'].values
    close_12h = df_12h['close'].values
    bullish_12h = close_12h > open_12h
    bearish_12h = close_12h < open_12h
    
    # Align 12h trend to 6h timeframe
    bullish_12h_aligned = align_htf_to_ltf(prices, df_12h, bullish_12h.astype(float))
    bearish_12h_aligned = align_htf_to_ltf(prices, df_12h, bearish_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(pp_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or \
           np.isnan(bullish_12h_aligned[i]) or np.isnan(bearish_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below Pivot Point
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above Pivot Point
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H4 level with 12h bullish candle
            if close[i] > h4_aligned[i] and bullish_12h_aligned[i] > 0.5:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L4 level with 12h bearish candle
            elif close[i] < l4_aligned[i] and bearish_12h_aligned[i] > 0.5:
                position = -1
                signals[i] = -0.25
    
    return signals