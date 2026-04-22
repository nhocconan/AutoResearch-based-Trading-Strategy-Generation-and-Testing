#!/usr/bin/env python3

"""
Hypothesis: Daily Donchian(20) breakout with weekly EMA34 trend filter and volume spike.
This strategy captures long-term breakouts aligned with weekly trend, filtered by volume
confirmation to avoid false breakouts. Works in both bull and bear markets by following
the weekly trend direction. Target: 15-25 trades/year per symbol.
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
    
    # Load weekly data - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Need to resample to daily first, but we'll use the prices data directly
    # Since we're on 1d timeframe, we can use the prices directly
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with bullish weekly trend and volume spike
            vol_avg_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.mean(volume[:i+1])
            if (close[i] > high_max_20[i] and 
                close[i] > ema34_1w_aligned[i] and  # Bullish trend: price above weekly EMA34
                volume[i] > 2.0 * vol_avg_20):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with bearish weekly trend and volume spike
            elif (close[i] < low_min_20[i] and 
                  close[i] < ema34_1w_aligned[i] and  # Bearish trend: price below weekly EMA34
                  volume[i] > 2.0 * vol_avg_20):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Donchian level
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to or below Donchian low
                if close[i] <= low_min_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to or above Donchian high
                if close[i] >= high_max_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_20_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0