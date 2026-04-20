#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h price data
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d Donchian channel (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels
    upper_1d = np.full_like(high_1d, np.nan)
    lower_1d = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        upper_1d[i] = np.max(high_1d[i-20:i])
        lower_1d[i] = np.min(low_1d[i-20:i])
    
    # Align to 12h timeframe
    upper_12h = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_12h = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # 1d ATR (14-period)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(low_1d[1:] - df_1d['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_1d[i] = np.nanmean(tr[1:15])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h volume average (20-period)
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN
        if (np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume spike
            if close[i] > upper_12h[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume spike
            elif close[i] < lower_12h[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: hold or exit on opposite break
            if close[i] < lower_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: hold or exit on opposite break
            if close[i] > upper_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeBreakout_v1"
timeframe = "12h"
leverage = 1.0