#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 levels from previous day
    # R1 = C + 1.1*(H-L)/12, S1 = C - 1.1*(H-L)/12
    R1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    S1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h timeframe (previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume filter: above average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        in_session = 8 <= hours[i] <= 20
        
        if position == 0:
            # Long: Close breaks above R1 + above 12h EMA + volume
            if (close[i] > R1_aligned[i] and 
                close[i] > ema_12h_aligned[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 + below 12h EMA + volume
            elif (close[i] < S1_aligned[i] and 
                  close[i] < ema_12h_aligned[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close breaks below S1
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close breaks above R1
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals