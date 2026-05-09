#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Volume_Trend"
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
    
    # Get 1d data for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume average
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 4h
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_1d_4h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_4h[i]) or np.isnan(vol_avg_1d_4h[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_1d_4h[i]
        vol_avg = vol_avg_1d_4h[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        if position == 0:
            # Long: break above upper Donchian with volume and above 1d EMA50
            if close[i] > upper[i] and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume and below 1d EMA50
            elif close[i] < lower[i] and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below lower Donchian or trend reversal
            if close[i] < lower[i] or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above upper Donchian or trend reversal
            if close[i] > upper[i] or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals