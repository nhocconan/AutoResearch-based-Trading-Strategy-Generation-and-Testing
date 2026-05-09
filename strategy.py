#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian_Breakout_Trend_Volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume average for volume filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly data to daily
    high_20_daily = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_daily = align_htf_to_ltf(prices, df_1w, low_20)
    ema50_1w_daily = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_daily[i]) or np.isnan(low_20_daily[i]) or 
            np.isnan(ema50_1w_daily[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = high_20_daily[i]
        lower = low_20_daily[i]
        trend = ema50_1w_daily[i]
        vol_ok = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: break above weekly Donchian upper with volume and above weekly EMA50
            if close[i] > upper and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian lower with volume and below weekly EMA50
            elif close[i] < lower and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below weekly Donchian lower or trend reversal
            if close[i] < lower or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above weekly Donchian upper or trend reversal
            if close[i] > upper or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals