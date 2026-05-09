#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Donchian20_Trend_Volume"
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
    
    # Get 1w data for trend (donchian channel breakout)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w Donchian channel (20-period)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # 1d volume filter: current volume > 1.5 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.5)
    
    # Align to 12h
    high_20_12h = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_12h = align_htf_to_ltf(prices, df_1w, low_20)
    volume_filter_12h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_12h[i]) or np.isnan(low_20_12h[i]) or
            np.isnan(volume_filter_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = high_20_12h[i]
        lower = low_20_12h[i]
        vol_filter = volume_filter_12h[i]
        
        if position == 0:
            # Enter long: break above upper Donchian with volume
            if close[i] > upper and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower Donchian with volume
            elif close[i] < lower and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below lower Donchian (mean reversion)
            if close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above upper Donchian (mean reversion)
            if close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals