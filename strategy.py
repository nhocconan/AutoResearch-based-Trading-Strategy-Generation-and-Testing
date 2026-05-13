#!/usr/bin/env python3
name = "1h_Donchian_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4H Donchian breakout for trend direction (upper/lower bands)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    upper_band = align_htf_to_ltf(prices, df_4h, high_20_4h)
    lower_band = align_htf_to_ltf(prices, df_4h, low_20_4h)
    
    # 1D volume filter (high volume confirms institutional interest)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = align_htf_to_ltf(prices, df_1d, volume_1d > vol_ma20_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if not session_ok[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above 4H upper band with volume spike
            if close[i] > upper_band[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Break below 4H lower band with volume spike
            elif close[i] < lower_band[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below 4H lower band or volume drops
            if close[i] < lower_band[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price rises back above 4H upper band or volume drops
            if close[i] > upper_band[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals