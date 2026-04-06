#!/usr/bin/env python3
"""
1h Donchian Breakout with 4h Trend Filter and Volume Confirmation v1
Hypothesis: 1h Donchian(20) breakouts filtered by 4h EMA50 trend and volume capture momentum.
Trades only during 08-20 UTC to avoid low-liquidity periods. Targets 60-150 trades over 4 years.
Works in bull (buy breakouts above 4h EMA50) and bear (sell breakouts below 4h EMA50).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian20_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_rising = ema50_4h > np.roll(ema50_4h, 1)
    ema50_falling = ema50_4h < np.roll(ema50_4h, 1)
    ema50_rising[0] = False
    ema50_falling[0] = False
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_4h, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_4h, ema50_falling)
    
    # 1h data
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
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 100  # For 4h EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = position * 0.20
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
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian band OR stoploss
            if (close[i] >= highest_high[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Donchian breakout + trend + volume
            bull_breakout = close[i] > highest_high[i]
            bear_breakout = close[i] < lowest_low[i]
            
            bull_entry = bull_breakout and ema50_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5
            bear_entry = bear_breakout and ema50_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
            if bull_entry:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals