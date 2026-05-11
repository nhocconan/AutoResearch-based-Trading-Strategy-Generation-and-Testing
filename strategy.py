#!/usr/bin/env python3
name = "1d_Weekly_Camarilla_R1S1_Breakout_Trend_Volume"
timeframe = "1d"
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
    
    # Weekly Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly Pivot calculation
    pivot_weekly = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    r1_weekly = pivot_weekly * 2 - df_1w['low']
    s1_weekly = pivot_weekly * 2 - df_1w['high']
    
    # Align Weekly Camarilla levels to daily timeframe
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_1w, pivot_weekly.values)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_1w, r1_weekly.values)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_1w, s1_weekly.values)
    
    # Weekly EMA 34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
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
        if (np.isnan(r1_weekly_aligned[i]) or np.isnan(s1_weekly_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close > R1 AND Volume > 1.5x MA AND Price > Weekly EMA34
            if (close[i] > r1_weekly_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 AND Volume > 1.5x MA AND Price < Weekly EMA34
            elif (close[i] < s1_weekly_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < S1 (reversal) OR Volume drops below average
            if (close[i] < s1_weekly_aligned[i] or volume[i] < 0.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Close > R1 (reversal) OR Volume drops below average
            if (close[i] > r1_weekly_aligned[i] or volume[i] < 0.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals