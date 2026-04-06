#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend and Volume Confirmation v2
Hypothesis: Donchian(20) breakouts on daily timeframe capture multi-week trends. 
Weekly EMA50 filters trend direction to avoid counter-trend trades. Volume confirms breakout strength.
Optimized for higher trade frequency (target: 75-250 total trades over 4 years) while maintaining robustness.
Works in bull (buy breakouts above) and bear (sell breakouts below) via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_weekly_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_prev = np.roll(ema50_weekly, 1)
    ema50_weekly_prev[0] = ema50_weekly[0]
    ema50_rising = ema50_weekly > ema50_weekly_prev
    ema50_falling = ema50_weekly < ema50_weekly_prev
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_weekly, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_weekly, ema50_falling)
    
    # Daily data
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
    start = 70  # For weekly EMA50 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema50_weekly_aligned[i]) or 
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i])):
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
            # Look for entries: Donchian breakout + trend + volume
            bull_breakout = close[i] > highest_high[i]
            bear_breakout = close[i] < lowest_low[i]
            
            bull_entry = bull_breakout and ema50_rising_aligned[i] and volume[i] > vol_ema[i] * 1.3
            bear_entry = bear_breakout and ema50_falling_aligned[i] and volume[i] > vol_ema[i] * 1.3
            
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