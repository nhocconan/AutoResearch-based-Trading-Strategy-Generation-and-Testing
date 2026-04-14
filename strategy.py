#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
# Donchian(20) breakout on daily captures major trend continuations
# Weekly ADX > 25 ensures we only trade in strong weekly trends
# Volume > 1.3x 20-day average confirms breakout strength
# Works in bull markets (breakouts up) and bear markets (breakouts down)
# Target: 10-25 trades/year per symbol to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ADX (14 periods)
    adx_len = 14
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate Donchian channels (20 periods)
    donch_len = 20
    upper_channel = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    lower_channel = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Calculate volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(60, donch_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly ADX > 25 indicates strong weekly trend
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout up + volume + weekly trend
            if (close[i] > upper_channel[i-1] and 
                volume_confirmed and 
                trending):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakout down + volume + weekly trend
            elif (close[i] < lower_channel[i-1] and 
                  volume_confirmed and 
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below midpoint of channel
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above midpoint of channel
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if close[i] > midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_WeeklyTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0