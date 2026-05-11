#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 levels (using previous day's data)
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        prev_close = close_1d[i-1]
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        range_val = prev_high - prev_low
        camarilla_r1[i] = prev_close + (range_val * 1.1 / 12)
        camarilla_s1[i] = prev_close - (range_val * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1-day EMA 34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1])
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close > Camarilla R1 AND Volume > 2.0x MA AND Price > 1d EMA34
            if (close[i] > camarilla_r1_aligned[i] and 
                volume[i] > 2.0 * vol_ma20[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < Camarilla S1 AND Volume > 2.0x MA AND Price < 1d EMA34
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume[i] > 2.0 * vol_ma20[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < Camarilla S1 OR Volume drops below average
            if (close[i] < camarilla_s1_aligned[i] or volume[i] < 0.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Close > Camarilla R1 OR Volume drops below average
            if (close[i] > camarilla_r1_aligned[i] or volume[i] < 0.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals