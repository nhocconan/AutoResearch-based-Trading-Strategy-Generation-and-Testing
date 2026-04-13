#!/usr/bin/env python3
"""
6h_1w_Donchian_Breakout_Weekly_Trend_Filter
Hypothesis: Weekly trend filters 6h Donchian breakouts to avoid counter-trend trades.
In bull markets (price above weekly SMA50), take long breakouts above 20-period high.
In bear markets (price below weekly SMA50), take short breakdowns below 20-period low.
Weekly trend reduces whipsaws during reversals. Volume confirmation ensures institutional participation.
Works in both bull and bear by adapting direction to weekly trend. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly SMA50 for trend filter
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(sma50_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: price above/below weekly SMA50
        weekly_uptrend = close[i] > sma50_1w_aligned[i]
        weekly_downtrend = close[i] < sma50_1w_aligned[i]
        
        # Donchian breakout signals
        long_breakout = close[i] > high_20[i] and volume_expansion[i]
        short_breakout = close[i] < low_20[i] and volume_expansion[i]
        
        # Enter long only in weekly uptrend
        if long_breakout and weekly_uptrend and position != 1:
            position = 1
            signals[i] = position_size
        # Enter short only in weekly downtrend
        elif short_breakout and weekly_downtrend and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit when trend changes or opposite signal
        elif position == 1 and not weekly_uptrend:
            position = 0
            signals[i] = 0.0
        elif position == -1 and not weekly_downtrend:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1w_Donchian_Breakout_Weekly_Trend_Filter"
timeframe = "6h"
leverage = 1.0