#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with 1d trend filter and volume confirmation.
Long when price breaks above Donchian upper channel with bullish 1d trend and volume spike.
Short when price breaks below Donchian lower channel with bearish 1d trend and volume spike.
Exit when price returns to Donchian middle channel.
Designed for low trade frequency (15-30/year) to minimize fee drag.
Works in both bull and bear markets via trend filter and volatility-based channels.
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
    
    # Load 1d data for trend filter and Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 1d EMA25 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema25_1d = close_1d.ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Align EMA25 to 12h timeframe
    ema25_aligned = align_htf_to_ltf(prices, df_1d, ema25_1d)
    
    # Calculate daily Donchian channels (20-period)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    
    # Donchian upper and lower channels (20-period high/low)
    high_max_20 = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    
    # Middle channel (average of upper and lower)
    middle_20 = (high_max_20 + low_min_20) / 2.0
    
    # Align all levels to 12h timeframe
    high_max_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    low_min_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle_20)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(25, n):  # Start after lookback
        # Skip if data not ready
        if (np.isnan(high_max_aligned[i]) or np.isnan(low_min_aligned[i]) or 
            np.isnan(middle_aligned[i]) or np.isnan(ema25_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
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
            # Long: Price breaks above Donchian upper with bullish 1d trend and volume spike
            if (close[i] > high_max_aligned[i] and 
                close[i] > ema25_aligned[i] and  # Bullish trend: price above EMA25
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with bearish 1d trend and volume spike
            elif (close[i] < low_min_aligned[i] and 
                  close[i] < ema25_aligned[i] and  # Bearish trend: price below EMA25
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Donchian middle
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle
                if close[i] <= middle_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle
                if close[i] >= middle_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_20_1dEMA25_Trend_Volume"
timeframe = "12h"
leverage = 1.0
#%%