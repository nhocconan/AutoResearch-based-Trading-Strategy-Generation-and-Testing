#!/usr/bin/env python3
"""
4H Donchian Channel Breakout with 1D Trend Filter and Volume Confirmation v1
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Using 1D EMA50 as trend filter
ensures we only trade in the direction of the higher timeframe trend, reducing false breakouts.
Volume confirmation ensures breakouts have conviction. Designed for 75-200 trades over 4 years
to minimize fee drag while adapting to bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1D data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1D EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 4H data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 50)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or trend reversal
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR trend turns bearish
            if low[i] <= lowest_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR trend turns bullish
            if high[i] >= highest_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume
            bull_breakout = high[i] >= highest_high[i]
            bear_breakout = low[i] <= lowest_low[i]
            bull_trend = close[i] > ema_50_aligned[i]
            bear_trend = close[i] < ema_50_aligned[i]
            vol_filter = volume[i] > vol_ma[i] * 1.5
            
            if bull_breakout and bull_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and bear_trend and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals