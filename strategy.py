#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + Volume + Weekly EMA Trend Filter
Hypothesis: Daily Donchian breakouts capture primary trends with low frequency. Volume confirms institutional participation. 
Weekly EMA filter ensures we only trade in the direction of the higher timeframe trend, reducing whipsaws. 
Designed for 30-100 trades over 4 years (7-25/year) to minimize fee drag. Works in both bull (breakouts above weekly EMA) and bear (breakdowns below weekly EMA) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_volume_weekly_ema_filter_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
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
        # Skip if required data not available
        if np.isnan(ema_weekly_aligned[i]):
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
            # Exit: price closes below Donchian lower OR price crosses below weekly EMA
            if close[i] < lowest_low or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR price crosses above weekly EMA
            if close[i] > highest_high or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly EMA trend filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Only go long if price is above weekly EMA, short if below
            trend_filter_long = close[i] > ema_weekly_aligned[i]
            trend_filter_short = close[i] < ema_weekly_aligned[i]
            
            if bull_breakout and volume_filter and trend_filter_long:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and volume_filter and trend_filter_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals