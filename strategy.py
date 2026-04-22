#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian channel breakout with volume confirmation and 1d trend filter.
Long when price breaks above 4h upper Donchian with bullish 1d trend (price > 1d EMA200) and volume spike.
Short when price breaks below 4h lower Donchian with bearish 1d trend (price < 1d EMA200) and volume spike.
Exit when price returns to 4h middle Donchian or trend reverses.
Target: 60-150 trades over 4 years (15-37/year) to minimize fee drag.
Uses 1h only for entry timing, 4h for signal direction, 1d for trend filter.
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
    
    # Load 4h data for Donchian channels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Align to 1h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_4h, mid_20)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMA200 to 1h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after lookbacks
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i]) or np.isnan(ema200_aligned[i]) or 
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
            # Long: Price breaks above 4h upper Donchian with bullish 1d trend and volume spike
            if (close[i] > upper_20_aligned[i] and 
                close[i] > ema200_aligned[i] and  # Bullish trend: price above 1d EMA200
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h lower Donchian with bearish 1d trend and volume spike
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < ema200_aligned[i] and  # Bearish trend: price below 1d EMA200
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to 4h middle Donchian OR trend turns bearish
                if close[i] <= mid_20_aligned[i] or close[i] < ema200_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to 4h middle Donchian OR trend turns bullish
                if close[i] >= mid_20_aligned[i] or close[i] > ema200_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Donchian_Breakout_1dEMA200_Trend_Volume"
timeframe = "1h"
leverage = 1.0