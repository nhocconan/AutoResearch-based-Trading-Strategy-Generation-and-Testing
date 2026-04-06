#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Trend Filter and Volume Confirmation v1
Hypothesis: Donchian(20) breakouts on 6h provide clear entry signals; weekly trend filter (price above/below weekly EMA20) ensures alignment with higher timeframe momentum; volume confirmation filters weak breakouts. Designed for 50-150 trades over 4 years to minimize fee drag while adapting to bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or trend reversal
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR weekly trend turns bearish
            if close[i] < lowest_low_20[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR weekly trend turns bullish
            if close[i] > highest_high_20[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly trend + volume
            bull_breakout = close[i] > highest_high_20[i]
            bear_breakout = close[i] < lowest_low_20[i]
            bull_trend = close[i] > ema_20_1w_aligned[i]
            bear_trend = close[i] < ema_20_1w_aligned[i]
            volume_ok = volume[i] > vol_ma[i] * 1.5
            
            if bull_breakout and bull_trend and volume_ok:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and bear_trend and volume_ok:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>