#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Index_Breakout_1dTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 4h data for Donchian breakout
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 1d choppiness index (14-period)
    high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(df_1d['high'] - df_1d['low']).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_14 / (high_14 - low_14)) / np.log10(14)
    
    # Calculate 1d EMA50 for trend
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    high_20_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align all to 4h
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, high_20_4h)
    low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, low_20_4h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data
    
    for i in range(start_idx, n):
        if (np.isnan(chop_4h[i]) or np.isnan(ema50_1d_4h[i]) or
            np.isnan(high_20_4h_aligned[i]) or np.isnan(low_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_4h[i]
        trend = ema50_1d_4h[i]
        upper_donchian = high_20_4h_aligned[i]
        lower_donchian = low_20_4h_aligned[i]
        
        # Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (breakout)
        if position == 0:
            # Enter long: break above upper Donchian in trending market
            if chop_val < 38.2 and close[i] > upper_donchian and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower Donchian in trending market
            elif chop_val < 38.2 and close[i] < lower_donchian and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: chop > 61.8 (ranging) or close below lower Donchian
            if chop_val > 61.8 or close[i] < lower_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: chop > 61.8 (ranging) or close above upper Donchian
            if chop_val > 61.8 or close[i] > upper_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals