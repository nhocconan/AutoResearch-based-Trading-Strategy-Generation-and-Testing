#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h EMA Trend + Volume Confirmation
Hypothesis: Donchian breakouts capture institutional breakout moves.
12h EMA filter ensures we trade only in the direction of higher timeframe trend.
Volume confirmation ensures breakouts have institutional participation.
Works in bull markets (breakouts to upside) and bear markets (breakouts to downside).
Target: 100-200 total trades over 4 years (~25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12h_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_prev = np.roll(ema50_12h, 1)
    ema50_12h_prev[0] = ema50_12h[0]
    ema50_rising = ema50_12h > ema50_12h_prev
    ema50_falling = ema50_12h < ema50_12h_prev
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema50_falling)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA50 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite breakout or stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss
            if (close[i] <= lowest_low[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss
            if (close[i] >= highest_high[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            long_breakout = close[i] > highest_high[i-1]  # Break above prior period high
            short_breakout = close[i] < lowest_low[i-1]   # Break below prior period low
            
            long_entry = long_breakout and ema50_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5
            short_entry = short_breakout and ema50_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals