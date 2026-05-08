#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels from previous day
    R1 = np.zeros(len(close_1d))
    S1 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        range_val = high_prev - low_prev
        
        C = close_prev + (range_val * 1.1 / 12)
        R1[i] = C + (range_val * 1.1 / 2)
        S1[i] = C - (range_val * 1.1 / 2)
    
    # Align Camarilla levels to daily timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, 4h uptrend, volume filter
            long_cond = (close[i] > R1_aligned[i] and 
                        ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and
                        volume_filter[i])
            
            # Short: price breaks below S1, 4h downtrend, volume filter
            short_cond = (close[i] < S1_aligned[i] and 
                         ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and
                         volume_filter[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below S1
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above R1
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals