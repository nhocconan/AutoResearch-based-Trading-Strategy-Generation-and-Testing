#!/usr/bin/env python3
# 6h_Donchian20_WeeklyPivotDir_Volume
# Hypothesis: Uses 6h Donchian breakout (20-period) with weekly pivot direction filter (derived from 1d data) and volume confirmation.
# Weekly pivot direction: if weekly close > weekly open → bullish bias (long bias); else bearish bias (short bias).
# Long when price breaks above Donchian upper + weekly bullish + volume spike.
# Short when price breaks below Donchian lower + weekly bearish + volume spike.
# Designed for 6h to achieve 50-150 trades over 4 years with clear trend-following logic that works in both bull and bear markets via weekly bias filter.

name = "6h_Donchian20_WeeklyPivotDir_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot direction calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate weekly bias from daily data: weekly bullish if weekly close > weekly open
    # Approximate weekly by checking if Friday's close > Monday's open (simplified proxy)
    # More robust: use 5-day rolling to simulate weekly
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).apply(lambda x: x[-1] if len(x) == 5 else np.nan, raw=True).values
    weekly_open = pd.Series(open_1d).rolling(window=5, min_periods=5).apply(lambda x: x[0] if len(x) == 5 else np.nan, raw=True).values
    weekly_bullish = weekly_close > weekly_open  # True if weekly bullish
    
    # Align weekly bias to 6h timeframe
    weekly_bullish_6h = align_htf_to_ltf(prices, df_1d, weekly_bullish.astype(float))
    
    # Calculate Donchian channels on 6h (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_bullish_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian upper + weekly bullish + volume spike
            if close[i] > highest_high[i] and weekly_bullish_6h[i] > 0.5 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower + weekly bearish + volume spike
            elif close[i] < lowest_low[i] and weekly_bullish_6h[i] < 0.5 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below Donchian lower or weekly bias turns bearish
            if close[i] < lowest_low[i] or weekly_bullish_6h[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above Donchian upper or weekly bias turns bullish
            if close[i] > highest_high[i] or weekly_bullish_6h[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals