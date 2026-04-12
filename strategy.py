#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_volume
Uses Donchian channel breakout on 4h with volume confirmation and 1d trend filter.
Long when price breaks above 20-period high with volume and 1d close > SMA50.
Short when price breaks below 20-period low with volume and 1d close < SMA50.
Exit when price crosses the 20-period opposite band.
Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag.
Works in both trending and ranging markets by combining breakout logic with trend filter.
"""

name = "4h_1d_donchian_breakout_volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Donchian Channel (20) on 4h
    dc_length = 20
    upper_dc = pd.Series(high).rolling(window=dc_length, min_periods=dc_length).max().values
    lower_dc = pd.Series(low).rolling(window=dc_length, min_periods=dc_length).min().values
    
    # 1d trend filter: SMA50
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Volume confirmation on 4h: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(sma50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above upper DC with volume and 1d trend up
        if close[i] > upper_dc[i] and vol_confirm[i] and close_1d[i // 24] > sma50_1d[i // 24] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below lower DC with volume and 1d trend down
        elif close[i] < lower_dc[i] and vol_confirm[i] and close_1d[i // 24] < sma50_1d[i // 24] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and close[i] < lower_dc[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > upper_dc[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals