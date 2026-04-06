#!/usr/bin/env python3
"""
6h Donchian breakout with weekly pivot direction and volume confirmation
Hypothesis: Donchian breakouts capture strong directional moves. Weekly pivot (from 1w) filters direction: only take long breaks above weekly pivot in uptrend, short breaks below in downtrend. Volume confirms institutional participation. Works in bull (breakouts up) and bear (breakouts down). Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1w_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot points: (H+L+C)/3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Weekly trend: higher highs and higher lows for uptrend
    # Simple: price above 20-period EMA = uptrend
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = close_1w > ema20_1w
    
    # Align to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        donchian_high[i] = np.max(high[i-lookback:i])
        donchian_low[i] = np.min(low[i-lookback:i])
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, lookback)  # For Donchian and volume EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(weekly_uptrend_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            atr_proxy = (high[i] - low[i])  # Simple range as ATR proxy
            if (close[i] <= donchian_low[i] or 
                close[i] <= entry_price - 2.0 * atr_proxy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            atr_proxy = (high[i] - low[i])
            if (close[i] >= donchian_high[i] or 
                close[i] >= entry_price + 2.0 * atr_proxy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot direction + volume
            atr_proxy = (high[i] - low[i])
            bull_breakout = close[i] > donchian_high[i]
            bear_breakout = close[i] < donchian_low[i]
            
            # Weekly pivot direction: only take longs in weekly uptrend, shorts in downtrend
            weekly_long_filter = weekly_uptrend_aligned[i]  # Only long when weekly uptrend
            weekly_short_filter = not weekly_uptrend_aligned[i]  # Only short when weekly downtrend
            
            vol_filter = volume[i] > vol_ema[i] * 1.5
            
            if bull_breakout and weekly_long_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and weekly_short_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals