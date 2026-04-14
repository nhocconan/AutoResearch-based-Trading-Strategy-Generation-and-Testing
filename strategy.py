#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ADX trend filter
# Uses 4h Donchian(20) breakouts with volume > 1.5x average volume
# Only takes breakouts in direction of 1d ADX trend (ADX > 25)
# Exit when price returns to Donchian midpoint or opposite breakout
# Works in both bull and bear markets: trend filter ensures we trade with higher timeframe trend
# Volume confirmation reduces false breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian channels (20 periods)
    donch_len = 20
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper and lower bands
    upper = pd.Series(high_4h).rolling(window=donch_len, min_periods=donch_len).max().values
    lower = pd.Series(low_4h).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Align Donchian to 4h timeframe (no additional alignment needed as we're already in 4h)
    # But we need to align to lower timeframe if we were using different timeframe
    # Since we're using 4h data for 4h signals, we can use directly
    upper_4h = upper
    lower_4h = lower
    
    # Load 1d data ONCE for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX (14 periods)
    adx_len = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
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
    
    # Align ADX to 4h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(donch_len, adx_len, 20) + 1
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_4h[i]) or 
            np.isnan(lower_4h[i]) or 
            np.isnan(adx_4h_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_4h_aligned[i] > 25
        
        # Volume confirmation: volume > 1.5x average
        vol_confirmed = vol > 1.5 * vol_avg[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + trend + volume
            if price > upper_4h[i] and trending and vol_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below lower Donchian + trend + volume
            elif price < lower_4h[i] and trending and vol_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR opposite breakout
            midpoint = (upper_4h[i] + lower_4h[i]) / 2
            if price < midpoint or price < lower_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR opposite breakout
            midpoint = (upper_4h[i] + lower_4h[i]) / 2
            if price > midpoint or price > upper_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Volume_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0