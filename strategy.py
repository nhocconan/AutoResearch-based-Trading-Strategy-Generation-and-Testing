#!/usr/bin/env python3
name = "1h_Hybrid_Camarilla_4D_1D_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4-hour Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for 4h
    camarilla_high_4h = np.full(len(close_4h), np.nan)
    camarilla_low_4h = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        day_high = high_4h[i-1]
        day_low = low_4h[i-1]
        day_close = close_4h[i-1]
        if np.isnan(day_high) or np.isnan(day_low) or np.isnan(day_close):
            continue
        camarilla_high_4h[i] = day_close + (day_high - day_low) * 1.1 / 12
        camarilla_low_4h[i] = day_close - (day_high - day_low) * 1.1 / 12
    
    camarilla_high_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_high_4h)
    camarilla_low_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_low_4h)
    
    # Daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    # Session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_high_4h_aligned[i]) or 
            np.isnan(camarilla_low_4h_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            if close[i] > camarilla_high_4h_aligned[i] and close[i] > ema_200_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.20
                position = 1
            elif close[i] < camarilla_low_4h_aligned[i] and close[i] < ema_200_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            if close[i] < camarilla_low_4h_aligned[i] or close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            if close[i] > camarilla_high_4h_aligned[i] or close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals