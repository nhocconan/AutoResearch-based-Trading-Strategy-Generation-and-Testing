#!/usr/bin/env python3
"""
1d Donchian breakout with weekly trend filter and volume confirmation
Hypothesis: Daily Donchian(20) breakouts capture significant momentum moves.
Weekly EMA20 filters trend direction to avoid counter-trend trades.
Volume confirms institutional participation. Works in both bull (breakout above upper band)
and bear (breakdown below lower band) markets.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA20 for trend filter
    close_weekly = df_weekly['close'].values
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_prev = np.roll(ema20_weekly, 1)
    ema20_weekly_prev[0] = ema20_weekly[0]
    ema20_rising = ema20_weekly > ema20_weekly_prev
    ema20_falling = ema20_weekly < ema20_weekly_prev
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    ema20_rising_aligned = align_htf_to_ltf(prices, df_weekly, ema20_rising)
    ema20_falling_aligned = align_htf_to_ltf(prices, df_weekly, ema20_falling)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    upper_channel = rolling_max(high, 20)
    lower_channel = rolling_min(low, 20)
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Donchian and weekly EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema20_weekly_aligned[i]) or 
            np.isnan(ema20_rising_aligned[i]) or np.isnan(ema20_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite breakout or stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower channel OR stoploss
            if (close[i] <= lower_channel[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper channel OR stoploss
            if (close[i] >= upper_channel[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            breakout_up = close[i] > upper_channel[i-1]  # Break above previous upper channel
            breakout_down = close[i] < lower_channel[i-1]  # Break below previous lower channel
            volume_filter = volume[i] > vol_ema[i] * 1.5
            
            if breakout_up and ema20_rising_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif breakout_down and ema20_falling_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals