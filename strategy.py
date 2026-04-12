#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Precompute hour filter for 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Supertrend (ATR=10, mult=3)
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_4h = np.full(len(df_4h), np.nan)
    for i in range(10, len(df_4h)):
        atr_4h[i] = np.mean(tr[i-10:i+1])
    
    upper = (high_4h + low_4h) / 2 + 3 * atr_4h
    lower = (high_4h + low_4h) / 2 - 3 * atr_4h
    
    supertrend = np.full(len(df_4h), np.nan)
    direction = np.full(len(df_4h), 1)  # 1 for up, -1 for down
    for i in range(10, len(df_4h)):
        if i == 10:
            supertrend[i] = upper[i]
            direction[i] = 1
        else:
            if close_4h[i-1] > supertrend[i-1]:
                upper[i] = min(upper[i], upper[i-1])
            else:
                lower[i] = max(lower[i], lower[i-1])
            
            if close_4h[i] > supertrend[i-1]:
                direction[i] = 1
            else:
                direction[i] = -1
            
            supertrend[i] = lower[i] if direction[i] == -1 else upper[i]
    
    # Align 4h Supertrend to 1h timeframe
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align 1d volume MA to 1h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(supertrend_4h_aligned[i]) or np.isnan(direction_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1h volume > 1.5 * 20-day average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Trend direction from 4h Supertrend
        uptrend = direction_4h_aligned[i] == 1
        downtrend = direction_4h_aligned[i] == -1
        
        # Entry conditions: price crosses Supertrend in trend direction + volume filter
        long_entry = (close[i] > supertrend_4h_aligned[i] and close[i-1] <= supertrend_4h_aligned[i-1]) and uptrend and vol_filter
        short_entry = (close[i] < supertrend_4h_aligned[i] and close[i-1] >= supertrend_4h_aligned[i-1]) and downtrend and vol_filter
        
        # Exit conditions: price crosses Supertrend in opposite direction
        long_exit = (close[i] < supertrend_4h_aligned[i] and close[i-1] >= supertrend_4h_aligned[i-1])
        short_exit = (close[i] > supertrend_4h_aligned[i] and close[i-1] <= supertrend_4h_aligned[i-1])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_supertrend_vol_filter_v1"
timeframe = "1h"
leverage = 1.0