#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on daily (price channel structure)
    high_20 = np.full(len(close_1d), np.nan)
    low_20 = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        high_20[i] = np.max(high_1d[i-20:i])
        low_20[i] = np.min(low_1d[i-20:i])
    
    # Calculate 10-period ATR on daily for volatility filter
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(10, len(tr_1d)):
        atr_1d[i] = np.mean(tr_1d[i-9:i+1])
    
    # Calculate 20-period average volume on daily for volume filter
    vol_ma_20 = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-20:i])
    
    # Align indicators to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = volume[i] > vol_ma_20_aligned[i] * 1.5
        
        # Donchian breakout conditions with volatility-adjusted thresholds
        upper_threshold = high_20_aligned[i] + 0.1 * atr_1d_aligned[i]
        lower_threshold = low_20_aligned[i] - 0.1 * atr_1d_aligned[i]
        
        long_breakout = close[i] > upper_threshold
        short_breakout = close[i] < lower_threshold
        
        # Entry conditions: volatility-filtered breakout
        long_entry = long_breakout and vol_filter
        short_entry = short_breakout and vol_filter
        
        # Exit conditions: opposite breakout
        exit_long = position == 1 and short_breakout
        exit_short = position == -1 and long_breakout
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_volatility_filter"
timeframe = "4h"
leverage = 1.0