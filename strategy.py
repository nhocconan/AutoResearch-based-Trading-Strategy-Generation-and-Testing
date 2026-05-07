#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
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
    open_time = pd.to_datetime(prices['open_time'])
    
    # Load 4h data ONCE for trend filter and 1d for Camarilla pivot
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 34 or len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1h = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1d Camarilla pivot levels from previous day (standard formula)
    c_high = df_1d['high'].values
    c_low = df_1d['low'].values
    c_close = df_1d['close'].values
    
    pivot = (c_high + c_low + c_close) / 3
    range_val = c_high - c_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align pivot levels to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_1d, r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1)
    pivot_1h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00-20:00 UTC
    session_mask = (open_time.hour >= 8) & (open_time.hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or 
            np.isnan(ema_34_1h[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(pivot_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not session_mask[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above R1 in 4h uptrend with volume
            if close[i] > r1_1h[i] and close[i] > ema_34_1h[i] and vol_condition:
                signals[i] = 0.20
                position = 1
            # Short: break below S1 in 4h downtrend with volume
            elif close[i] < s1_1h[i] and close[i] < ema_34_1h[i] and vol_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price returns to pivot or trend reverses
            if close[i] < pivot_1h[i] or close[i] < ema_34_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price returns to pivot or trend reverses
            if close[i] > pivot_1h[i] or close[i] > ema_34_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals