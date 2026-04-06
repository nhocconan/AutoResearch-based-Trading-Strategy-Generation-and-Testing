#!/usr/bin/env python3
"""
1h Donchian Breakout with 4h Trend Filter and Volume Confirmation v1
Hypothesis: Donchian breakouts capture trend momentum; 4h EMA filter ensures alignment with higher timeframe trend; volume confirms breakout strength. Designed for 60-150 trades over 4 years (15-37/year) to minimize fee drag while working in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian_breakout_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA for trend direction
    ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_4h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or trend reversal
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR trend turns bearish
            if close[i] < lowest_low[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR trend turns bullish
            if close[i] > highest_high[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Donchian breakout + trend alignment + volume
            bull_breakout = close[i] > highest_high[i]
            bear_breakout = close[i] < lowest_low[i]
            trend_bullish = close[i] > ema_4h_aligned[i]
            trend_bearish = close[i] < ema_4h_aligned[i]
            volume_confirm = volume[i] > vol_ma[i] * 1.5
            
            if bull_breakout and trend_bullish and volume_confirm:
                signals[i] = 0.20
                position = 1
            elif bear_breakout and trend_bearish and volume_confirm:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals