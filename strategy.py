#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 1d ATR (14-period) for volatility filter
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.abs(high_1d[1:] - close_1d[:-1]), 
                       np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is reasonable
        vol_filter = atr_1d_aligned[i] > 0
        
        # Volume filter: current volume above 1d average
        volume_filter = volume[i] > vol_ma_1d_aligned[i]
        
        # Long condition: close above 1d Donchian high + volume + volatility
        long_condition = (close[i] > donchian_high_aligned[i] and 
                         volume_filter and 
                         vol_filter)
        
        # Short condition: close below 1d Donchian low + volume + volatility
        short_condition = (close[i] < donchian_low_aligned[i] and 
                          volume_filter and 
                          vol_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: price crosses back through the Donchian channel
        elif position == 1 and close[i] < donchian_low_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donchian_high_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_VolumeFilter"
timeframe = "4h"
leverage = 1.0