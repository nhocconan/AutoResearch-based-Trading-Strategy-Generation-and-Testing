#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: On 6h timeframe, Donchian breakouts aligned with weekly pivot bias (from 1w high/low) capture strong momentum.
Weekly pivot bias: price > weekly midpoint = bullish bias, price < weekly midpoint = bearish bias.
Volume confirms breakout strength. Designed for 50-150 trades over 4 years to minimize fee drag.
Works in bull (buy breakouts above with bullish bias) and bear (sell breakouts below with bearish bias).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1w_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data for pivot bias (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly high, low, close for pivot calculations
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points: P = (H + L + C)/3
    # Bias: price > P = bullish, price < P = bearish
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    bullish_bias = close_1w > pivot_1w  # weekly close above pivot = bullish bias
    bearish_bias = close_1w < pivot_1w  # weekly close below pivot = bearish bias
    
    # Align weekly bias to 6h timeframe
    bullish_bias_aligned = align_htf_to_ltf(prices, df_1w, bullish_bias)
    bearish_bias_aligned = align_htf_to_ltf(prices, df_1w, bearish_bias)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # Ensure weekly data alignment
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(bullish_bias_aligned[i]) or 
            np.isnan(bearish_bias_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite breakout or stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian band OR stoploss
            if (close[i] <= lowest_low[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian band OR stoploss
            if (close[i] >= highest_high[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot bias + volume
            bull_breakout = close[i] > highest_high[i]
            bear_breakout = close[i] < lowest_low[i]
            
            bull_entry = bull_breakout and bullish_bias_aligned[i] and volume[i] > vol_ema[i] * 1.5
            bear_entry = bear_breakout and bearish_bias_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals