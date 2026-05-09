#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Donchian20_WeeklyTrend_VolumeBreakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend (Higher Time Frame)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get weekly volume average for volume filter
    df_1w_vol = get_htf_data(prices, '1w')
    if len(df_1w_vol) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly volume average for volume filter
    vol_series_1w = pd.Series(df_1w_vol['volume'])
    vol_avg_1w = vol_series_1w.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channel on 1d (primary timeframe)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align all to 1d
    ema50_1w_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_avg_1w_1d = align_htf_to_ltf(prices, df_1w_vol, vol_avg_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(lookback, 50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_1d[i]) or np.isnan(vol_avg_1w_1d[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_1w_1d[i]
        vol_avg = vol_avg_1w_1d[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        if position == 0:
            # Long: break above Donchian high with volume and above weekly trend
            if high[i] > highest_high[i] and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and below weekly trend
            elif low[i] < lowest_low[i] and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below Donchian low or trend reversal
            if close[i] < lowest_low[i] or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above Donchian high or trend reversal
            if close[i] > highest_high[i] or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals