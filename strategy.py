#!/usr/bin/env python3
"""
4h_WeekendBreakout_1dTrend_Volume
Hypothesis: Breakouts from 1d Donchian channels (20-period) on weekends (Saturday-Sunday UTC) with volume confirmation and filtered by 1d EMA50 trend. Captures reduced liquidity breakouts during low-volume weekend sessions, which often exhibit stronger follow-through in both bull and bear markets. Designed for 4h timeframe to maintain low trade frequency (~20-40/year) while capturing high-probability moves.
"""

name = "4h_WeekendBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    upper_channel = align_htf_to_ltf(prices, df_1d, high_max_20)
    lower_channel = align_htf_to_ltf(prices, df_1d, low_min_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute weekend hours (Saturday=5, Sunday=6 in UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    days = pd.DatetimeIndex(prices["open_time"]).dayofweek  # Monday=0, Sunday=6
    is_weekend = (days == 5) | (days == 6)  # Saturday or Sunday
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above 1d upper channel + volume spike + 1d uptrend + weekend
            if close[i] > upper_channel[i] and vol_spike and close[i] > ema_50_1d_aligned[i] and is_weekend[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 1d lower channel + volume spike + 1d downtrend + weekend
            elif close[i] < lower_channel[i] and vol_spike and close[i] < ema_50_1d_aligned[i] and is_weekend[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 1d lower channel or trend reverses
            if close[i] < lower_channel[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 1d upper channel or trend reverses
            if close[i] > upper_channel[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals