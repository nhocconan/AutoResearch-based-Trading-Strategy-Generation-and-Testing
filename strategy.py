#!/usr/bin/env python3
name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h trend: EMA21 (faster for 1h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d volume filter: volume > 1.5 x 20-day average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 1h Camarilla R1/S1
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    pivot_1h = (high_1h + low_1h + close_1h) / 3
    range_1h = high_1h - low_1h
    r1_1h = close_1h + (range_1h * 1.0833)
    s1_1h = close_1h - (range_1h * 1.0833)
    r1_1h_aligned = align_htf_to_ltf(prices, df_1h, r1_1h)
    s1_1h_aligned = align_htf_to_ltf(prices, df_1h, s1_1h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if np.isnan(ema_21_4h_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or \
           np.isnan(r1_1h_aligned[i]) or np.isnan(s1_1h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R1, above 4h EMA21, volume above 1.5x 20-day avg
            if close[i] > r1_1h_aligned[i] and close[i] > ema_21_4h_aligned[i] and volume[i] > 1.5 * vol_ma20_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Close breaks below S1, below 4h EMA21, volume above 1.5x 20-day avg
            elif close[i] < s1_1h_aligned[i] and close[i] < ema_21_4h_aligned[i] and volume[i] > 1.5 * vol_ma20_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Close below S1 or below 4h EMA21
            if close[i] < s1_1h_aligned[i] or close[i] < ema_21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Close above R1 or above 4h EMA21
            if close[i] > r1_1h_aligned[i] or close[i] > ema_21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals