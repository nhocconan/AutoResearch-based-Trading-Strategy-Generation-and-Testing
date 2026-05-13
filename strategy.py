#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1dTrend_Volume
Hypothesis: Use 12-hour Donchian channel (20-period) breakouts with daily trend filter (price above/below EMA50) and volume confirmation (volume > 1.5x 20-period average). Go long when price breaks above upper Donchian band with daily uptrend and volume confirmation, short when price breaks below lower band with daily downtrend and volume confirmation. Exit on opposite Donchian break. Designed for 12h timeframe to limit trades (target 50-150 over 4 years) and avoid fee drag while capturing medium-term trends in both bull and bear markets.
"""

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period Donchian channels on 12h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: price breaks above upper Donchian + daily uptrend + volume confirmation
            if high[i] > high_max_20[i] and close[i] > ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian + daily downtrend + volume confirmation
            elif low[i] < low_min_20[i] and close[i] < ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below lower Donchian
            if low[i] < low_min_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above upper Donchian
            if high[i] > high_max_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals