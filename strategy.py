#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + Volume + Weekly EMA Filter
Hypothesis: Daily Donchian breakouts capture medium-term momentum. Volume confirms institutional participation. Weekly EMA filter ensures alignment with longer-term trend, reducing whipsaws. Designed for 30-100 trades over 4 years (7-25/year) to minimize fee drag. Works in bull (breakouts) and bear (breakdowns) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_volume_weekly_ema_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load weekly data for EMA (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(21)
    weekly_close = df_weekly['close'].values
    ema_21 = np.full_like(weekly_close, np.nan)
    if len(weekly_close) >= 21:
        ema_21[20] = np.mean(weekly_close[:21])
        for i in range(21, len(weekly_close)):
            ema_21[i] = (weekly_close[i] * 2/22) + (ema_21[i-1] * 20/22)
    
    # Align weekly EMA to daily
    ema_21_aligned = align_htf_to_ltf(prices, df_weekly, ema_21)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_21_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR price below weekly EMA
            if close[i] < lowest_low or close[i] < ema_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR price above weekly EMA
            if close[i] > highest_high or close[i] > ema_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly EMA filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            ema_filter_long = close[i] > ema_21_aligned[i]  # Above weekly EMA for long
            ema_filter_short = close[i] < ema_21_aligned[i]  # Below weekly EMA for short
            
            if bull_breakout and volume_filter and ema_filter_long:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and volume_filter and ema_filter_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals