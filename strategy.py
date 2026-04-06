#!/usr/bin/env python3
"""
1h Donchian(20) Breakout + 4h EMA Trend + Volume Filter
Hypothesis: On 1h timeframe, Donchian breakouts combined with 4h EMA trend filter and volume confirmation 
capture significant moves while maintaining low trade frequency. 4h EMA filter ensures we only trade 
in the direction of the higher timeframe trend, reducing whipsaws in both bull and bear markets.
Session filter (08-20 UTC) reduces noise trades.
Target: 60-150 total trades over 4 years (15-37/year) with low frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian20_4hma_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for EMA (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 20-period EMA on 4h
    ema_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 20:
        multiplier = 2 / (20 + 1)
        ema_4h[0] = close_4h[0]
        for i in range(1, len(close_4h)):
            ema_4h[i] = (close_4h[i] * multiplier) + (ema_4h[i-1] * (1 - multiplier))
    
    # Align EMA to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and EMA
    
    for i in range(start, n):
        # Skip if outside session or required data not available
        if not in_session[i] or np.isnan(ema_4h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR price below 4h EMA
            if close[i] < lowest_low or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR price above 4h EMA
            if close[i] > highest_high or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Donchian breakout + volume + 4h EMA trend filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Only go long if price above 4h EMA, short if below
            trend_uptrend = close[i] > ema_4h_aligned[i]
            trend_downtrend = close[i] < ema_4h_aligned[i]
            
            if i >= 20 and bull_breakout and volume_filter and trend_uptrend:
                signals[i] = 0.20
                position = 1
            elif i >= 20 and bear_breakout and volume_filter and trend_downtrend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals