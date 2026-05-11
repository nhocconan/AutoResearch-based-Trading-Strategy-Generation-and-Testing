#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_R1S1_Breakout_TrendFilter
Hypothesis: Uses 1h price breaking above/below Camarilla R1/S1 levels as entry signals, filtered by 4h EMA50 trend and 1d volume spike. Exits when price reverts to daily open or trend reverses. Designed for low trade frequency (<50/year) with clear signals in both bull and bear markets by following the higher timeframe trend.
"""

name = "1h_4h1d_Camarilla_R1S1_Breakout_TrendFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # 4h data for trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h Camarilla levels (R1, S1)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla calculation: (High - Low) * 1.1 / 12
    r1_4h = close_4h + (high_4h - low_4h) * 1.1 / 12
    s1_4h = close_4h - (high_4h - low_4h) * 1.1 / 12
    
    # Align Camarilla levels to 1h
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # 1d volume for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / vol_ma_1d
    vol_ratio_1d = np.nan_to_num(vol_ratio_1d, nan=1.0)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            not in_session[i]):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R1 + above 4h EMA50 + volume spike + in session
            if (close[i] > r1_4h_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + below 4h EMA50 + volume spike + in session
            elif (close[i] < s1_4h_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to daily open OR trend turns down
                if (close[i] <= open_price[i]) or \
                   (close[i] < ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price returns to daily open OR trend turns up
                if (close[i] >= open_price[i]) or \
                   (close[i] > ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals