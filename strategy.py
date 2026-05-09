#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Chop_Filter_Donchian_Breakout_1dTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 4h data for chop filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1d EMA100 for trend
    ema100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # 4h ATR14 for chop filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = np.maximum(high_4h - low_4h, np.abs(high_4h - np.roll(close_4h, 1)), np.abs(low_4h - np.roll(close_4h, 1)))
    tr1[0] = high_4h[0] - low_4h[0]
    atr14_4h = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # 4h Donchian channels (20)
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h True Range for chop calculation
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr1_4h = pd.Series(tr).rolling(window=1, min_periods=1).mean().values
    
    # Choppiness Index (14) - range bound indicator
    sum_atr14 = pd.Series(atr1_4h).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (max_hh - min_ll)) / np.log10(14)
    
    # Align all to 4h
    ema100_1d_4h = align_htf_to_ltf(prices, df_1d, ema100_1d)
    atr14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr14_4h)
    donch_high_20_4h = donch_high_20  # already 4h
    donch_low_20_4h = donch_low_20    # already 4h
    chop_4h = chop  # already 4h
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(ema100_1d_4h[i]) or np.isnan(atr14_4h_aligned[i]) or 
            np.isnan(donch_high_20_4h[i]) or np.isnan(donch_low_20_4h[i]) or 
            np.isnan(chop_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema100_1d_4h[i]
        atr14 = atr14_4h_aligned[i]
        upper = donch_high_20_4h[i]
        lower = donch_low_20_4h[i]
        chop_val = chop_4h[i]
        
        # Chop filter: only trade when chop < 38.2 (trending)
        trending = chop_val < 38.2
        
        if position == 0:
            # Long: break above Donchian high with trend alignment and trending market
            if close[i] > upper and close[i] > trend and trending:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with trend alignment and trending market
            elif close[i] < lower and close[i] < trend and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below Donchian low or trend reversal
            if close[i] < lower or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above Donchian high or trend reversal
            if close[i] > upper or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals