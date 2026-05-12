#!/usr/bin/env python3
"""
6h_WeeklyPivot_PriceChannelBreakout_VolumeSpike_HT
Hypothesis: Breakout above/below 6h price channel (20-period Donchian) with weekly pivot (Monday open) direction filter and volume confirmation (2x average) captures momentum in trending markets while avoiding chop. Weekly pivot provides directional bias: above = long bias, below = short bias. Works in bull/bear by following weekly pivot position relative to price.
"""

name = "6h_WeeklyPivot_PriceChannelBreakout_VolumeSpike_HT"
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
    open_time = prices['open_time'].values

    # Get weekly data (resample to weekly using Monday as start)
    # We'll compute weekly pivot from prior week's open (Monday)
    # For simplicity, use 1d data and aggregate to weekly manually
    df_1d = get_htf_data(prices, '1d')
    
    # Convert to DataFrame for resampling
    df_1d_df = pd.DataFrame({
        'open': df_1d['open'],
        'high': df_1d['high'],
        'low': df_1d['low'],
        'close': df_1d['close'],
        'volume': df_1d['volume']
    }, index=pd.to_datetime(df_1d['open_time']))
    
    # Resample to weekly, starting Monday
    df_weekly = df_1d_df.resample('W-MON').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Weekly pivot: use prior week's open (Monday open) as bias
    weekly_open = df_weekly['open'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to use prior week's data
    prev_weekly_open = np.roll(weekly_open, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_open[0] = np.nan
    prev_weekly_close[0] = np.nan
    
    # Bias: 1 if current week's open > prior week's close (bullish), -1 if bearish
    weekly_bias = np.where(prev_weekly_open > prev_weekly_close, 1, -1)
    weekly_bias = np.roll(weekly_bias, 1)  # Align to current week
    weekly_bias[0] = np.nan
    
    # Align weekly bias to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bias)
    
    # 6h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(weekly_bias_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + weekly bullish bias + volume spike
            if (close[i] > high_roll[i] and 
                weekly_bias_aligned[i] == 1 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + weekly bearish bias + volume spike
            elif (close[i] < low_roll[i] and 
                  weekly_bias_aligned[i] == -1 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian low
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian high
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals