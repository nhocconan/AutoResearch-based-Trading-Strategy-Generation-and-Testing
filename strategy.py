#!/usr/bin/env python3
"""
1d Donchian channel breakout with weekly trend filter and volume confirmation
Hypothesis: Buy when price breaks above weekly Donchian high in uptrend, sell when breaks below weekly Donchian low in downtrend.
Weekly trend defined by price above/below weekly EMA50. Volume confirms breakout strength.
Works in bull (buy breakouts) and bear (sell breakdowns). Target: 50-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA50 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema50_weekly_aligned[i]) or 
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly trend + volume
            bull_breakout = (close[i] > donchian_high[i] and 
                           ema50_rising_aligned[i] and 
                           volume[i] > vol_ema[i] * 1.5)
            bear_breakout = (close[i] < donchian_low[i] and 
                           ema50_falling_aligned[i] and 
                           volume[i] > vol_ema[i] * 1.5)
            
            if bull_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals