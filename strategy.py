#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot levels using previous week's data
    # Weekly pivot: P = (PH + PL + PC) / 3
    # Support levels: S1 = 2*P - PH, S2 = P - (PH - PL)
    # Resistance levels: R1 = 2*P - PL, R2 = P + (PH - PL)
    n1d = len(close_1d)
    weekly_P = np.full(n1d, np.nan)
    weekly_S1 = np.full(n1d, np.nan)
    weekly_S2 = np.full(n1d, np.nan)
    weekly_R1 = np.full(n1d, np.nan)
    weekly_R2 = np.full(n1d, np.nan)
    
    for i in range(1, n1d):
        PH = high_1d[i-1]  # Previous week high
        PL = low_1d[i-1]   # Previous week low
        PC = close_1d[i-1] # Previous week close
        
        P = (PH + PL + PC) / 3.0
        weekly_P[i] = P
        weekly_S1[i] = 2 * P - PH
        weekly_S2[i] = P - (PH - PL)
        weekly_R1[i] = 2 * P - PL
        weekly_R2[i] = P + (PH - PL)
    
    # Align weekly pivot levels to 12h timeframe
    weekly_P_aligned = align_htf_to_ltf(prices, df_1d, weekly_P)
    weekly_S1_aligned = align_htf_to_ltf(prices, df_1d, weekly_S1)
    weekly_S2_aligned = align_htf_to_ltf(prices, df_1d, weekly_S2)
    weekly_R1_aligned = align_htf_to_ltf(prices, df_1d, weekly_R1)
    weekly_R2_aligned = align_htf_to_ltf(prices, df_1d, weekly_R2)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(weekly_P_aligned[i]) or np.isnan(weekly_S1_aligned[i]) or 
            np.isnan(weekly_S2_aligned[i]) or np.isnan(weekly_R1_aligned[i]) or 
            np.isnan(weekly_R2_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 1w uptrend + volume spike
            long_cond = (close[i] > weekly_R1_aligned[i] and 
                        ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below S1 with 1w downtrend + volume spike
            short_cond = (close[i] < weekly_S1_aligned[i] and 
                         ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below P (pivot point)
            if close[i] < weekly_P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above P (pivot point)
            if close[i] > weekly_P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals