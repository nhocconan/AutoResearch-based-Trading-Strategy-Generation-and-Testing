#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with daily Donchian breakout + weekly trend filter + volume confirmation
# Donchian breakout captures trending moves, weekly filter ensures alignment with higher timeframe trend,
# volume confirmation reduces false breakouts. Designed for low trade frequency to avoid fee drag.
# Works in bull markets (breakouts) and bear markets (breakdowns with trend filter).
name = "12h_Donchian20_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (20-day high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels on daily data
    highest_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for daily close)
    highest_20_12h = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_12h = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Get 1w data for weekly trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to 12h timeframe
    ema_34_1w_12h = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: above 1.5x 24-period average (24*12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20_12h[i]) or np.isnan(lowest_20_12h[i]) or 
            np.isnan(ema_34_1w_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above 20-day high with weekly uptrend
            if (close[i] > highest_20_12h[i] and 
                close[i] > ema_34_1w_12h[i] and  # Weekly uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below 20-day low with weekly downtrend
            elif (close[i] < lowest_20_12h[i] and 
                  close[i] < ema_34_1w_12h[i] and  # Weekly downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-day low (mean reversion)
            if close[i] < lowest_20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-day high (mean reversion)
            if close[i] > highest_20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals