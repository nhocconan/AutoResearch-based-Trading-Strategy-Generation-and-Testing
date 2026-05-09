#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-day high/low)
    high_series = pd.Series(df_1d['high'].values)
    low_series = pd.Series(df_1d['low'].values)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Trend filter: 12h EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 12h volume > 1.5 * 20-day average
    vol_series = pd.Series(df_12h['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = df_12h['volume'].values > (vol_ma * 1.5)
    
    # Align all to 4h
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema50_12h_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    volume_filter_4h = align_htf_to_ltf(prices, df_12h, volume_filter_12h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or
            np.isnan(ema50_12h_4h[i]) or np.isnan(volume_filter_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = donchian_high_4h[i]
        lower = donchian_low_4h[i]
        trend = ema50_12h_4h[i]
        vol_filter = volume_filter_4h[i]
        
        if position == 0:
            # Enter long: break above upper Donchian with volume and above trend
            if close[i] > upper and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower Donchian with volume and below trend
            elif close[i] < lower and close[i] < trend and vol_filter:
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