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
    
    # Get daily data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on daily data
    period = 20
    upper_1d = np.full(len(high_1d), np.nan)
    lower_1d = np.full(len(low_1d), np.nan)
    
    for i in range(period-1, len(high_1d)):
        upper_1d[i] = np.max(high_1d[i-period+1:i+1])
        lower_1d[i] = np.min(low_1d[i-period+1:i+1])
    
    # Calculate ATR on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr_period = 14
    atr_1d = np.full(len(tr), np.nan)
    for i in range(atr_period-1, len(tr)):
        if i == atr_period-1:
            atr_1d[i] = np.mean(tr[0:i+1])
        else:
            atr_1d[i] = (atr_1d[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Align daily indicators to 6h timeframe
    upper_6h = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_6h = align_htf_to_ltf(prices, df_1d, lower_1d)
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate volume moving average (20-period on 6h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 20)  # need Donchian, ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above upper Donchian with volume
            if (close[i] > upper_6h[i] and vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian with volume
            elif (close[i] < lower_6h[i] and vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price closes below lower Donchian or ATR-based stop
            if (close[i] < lower_6h[i] or 
                close[i] < (upper_6h[i] - 2.0 * atr_6h[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above upper Donchian or ATR-based stop
            if (close[i] > upper_6h[i] or 
                close[i] > (lower_6h[i] + 2.0 * atr_6h[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_Volume_ATRStop"
timeframe = "6h"
leverage = 1.0