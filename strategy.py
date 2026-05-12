#!/usr/bin/env python3
"""
6H_DONCHIAN_BREAKOUT_20_WEEKLY_DIRECTION_1D_TREND_FILTER
Hypothesis: 6-hour Donchian(20) breakouts aligned with weekly trend direction from daily candles
provide high-probability entries in both bull and bear markets. Weekly trend is determined by
price position relative to 20-week EMA on daily timeframe (proxy for weekly trend). Volume
confirmation filters false breakouts. Target: 25-40 trades/year on 6h timeframe.
"""
name = "6H_DONCHIAN_BREAKOUT_20_WEEKLY_DIRECTION_1D_TREND_FILTER"
timeframe = "6h"
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
    
    # Daily data for weekly trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 20-period EMA on daily (proxy for weekly trend)
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Weekly trend: 1 = uptrend (price above EMA20), -1 = downtrend (price below EMA20)
    weekly_trend = np.where(close_1d > ema20_1d, 1, -1)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1d, weekly_trend)
    
    # Donchian channels (20-period) on 6h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Need Donchian formation
        # Skip if weekly trend not available
        if np.isnan(weekly_trend_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with volume in uptrend
            if (high[i] > high_roll[i-1] and 
                volume_spike[i] and 
                weekly_trend_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with volume in downtrend
            elif (low[i] < low_roll[i-1] and 
                  volume_spike[i] and 
                  weekly_trend_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or trend reversal
            if (low[i] < low_roll[i-1] or 
                weekly_trend_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or trend reversal
            if (high[i] > high_roll[i-1] or 
                weekly_trend_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals