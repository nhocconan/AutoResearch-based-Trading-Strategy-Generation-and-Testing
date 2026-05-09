#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_TopBottom_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian breakout levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1w Donchian channels (20-period high/low)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume average for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h timeframe
    high_20_12h = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_12h = align_htf_to_ltf(prices, df_1w, low_20)
    ema50_1d_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_1d_12h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_12h[i]) or np.isnan(low_20_12h[i]) or 
            np.isnan(ema50_1d_12h[i]) or np.isnan(vol_avg_1d_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = high_20_12h[i]
        lower = low_20_12h[i]
        trend = ema50_1d_12h[i]
        vol_avg = vol_avg_1d_12h[i]
        vol_ok = volume[i] > vol_avg * 1.8
        
        if position == 0:
            # Long: break above 20-week high with volume and above 1d EMA50
            if close[i] > upper and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-week low with volume and below 1d EMA50
            elif close[i] < lower and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below 20-week low or trend reversal
            if close[i] < lower or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above 20-week high or trend reversal
            if close[i] > upper or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals