#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Pivot from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily Pivot calculation
    pivot_daily = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    r1_daily = pivot_daily * 2 - df_1d['low']
    s1_daily = pivot_daily * 2 - df_1d['high']
    
    # Align Pivot levels to 12h timeframe
    pivot_daily_aligned = align_htf_to_ltf(prices, df_1d, pivot_daily.values)
    r1_daily_aligned = align_htf_to_ltf(prices, df_1d, r1_daily.values)
    s1_daily_aligned = align_htf_to_ltf(prices, df_1d, s1_daily.values)
    
    # Daily EMA 34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1])
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(r1_daily_aligned[i]) or np.isnan(s1_daily_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close > R1 AND Volume > 1.5x MA AND Price > Daily EMA34
            if (close[i] > r1_daily_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 AND Volume > 1.5x MA AND Price < Daily EMA34
            elif (close[i] < s1_daily_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < S1 (reversal) OR Volume drops below average
            if (close[i] < s1_daily_aligned[i] or volume[i] < 0.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Close > R1 (reversal) OR Volume drops below average
            if (close[i] > r1_daily_aligned[i] or volume[i] < 0.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals