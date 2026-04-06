#!/usr/bin/env python3
"""
4h Donchian Breakout with 12h Trend Filter and Volume Spike v1
Hypothesis: Donchian(20) breakouts on 4h timeframe capture strong momentum moves.
12h EMA(50) filter ensures trades align with higher timeframe trend.
Volume spike (2x 20-period average) confirms breakout strength.
Designed for 75-200 trades over 4 years to minimize fee drag while adapting to bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 50)  # For Donchian and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite breakout or trend reversal
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR trend turns bearish
            if close[i] < lowest_low_20[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR trend turns bullish
            if close[i] > highest_high_20[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout + trend alignment + volume spike
            bull_breakout = close[i] > highest_high_20[i]
            bear_breakout = close[i] < lowest_low_20[i]
            bull_trend = close[i] > ema_50_12h_aligned[i]
            bear_trend = close[i] < ema_50_12h_aligned[i]
            volume_spike = volume[i] > vol_ma[i] * 2.0
            
            if bull_breakout and bull_trend and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and bear_trend and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals