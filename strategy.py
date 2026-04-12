#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_donchian_breakout_volume"
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
    
    # Get 4h data for Donchian channel (trend filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channel (20-period) on 4h data
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    upper_4h = high_4h_series.rolling(window=20, min_periods=20).max().values
    lower_4h = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_1d_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_ma)
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    # 1h volume filter (short-term)
    volume_1h_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(volume_1d_ma_aligned[i]) or np.isnan(volume_1h_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not session_ok[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume filters: 1h volume > 1h MA AND 1d volume > 1d MA
        volume_ok = volume[i] > volume_1h_ma[i] and volume_1d_ma_aligned[i] > 0 and \
                   volume_1d.iloc[i // 16] > volume_1d_ma_aligned[i] if i >= 16 else False
        
        # Donchian breakout conditions
        breakout_long = close[i] > upper_4h_aligned[i]
        breakout_short = close[i] < lower_4h_aligned[i]
        
        # Exit conditions: opposite Donchian level
        exit_long = close[i] < lower_4h_aligned[i]
        exit_short = close[i] > upper_4h_aligned[i]
        
        # Execute trades
        if breakout_long and volume_ok and position != 1:
            position = 1
            signals[i] = 0.20
        elif breakout_short and volume_ok and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals