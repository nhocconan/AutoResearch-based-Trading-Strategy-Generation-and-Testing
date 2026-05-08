#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels
    n1w = len(close_1w)
    weekly_P = np.full(n1w, np.nan)
    weekly_S1 = np.full(n1w, np.nan)
    weekly_S2 = np.full(n1w, np.nan)
    weekly_R1 = np.full(n1w, np.nan)
    weekly_R2 = np.full(n1w, np.nan)
    
    for i in range(1, n1w):
        PH = high_1w[i-1]
        PL = low_1w[i-1]
        PC = close_1w[i-1]
        
        P = (PH + PL + PC) / 3.0
        weekly_P[i] = P
        weekly_S1[i] = 2 * P - PH
        weekly_S2[i] = P - (PH - PL)
        weekly_R1[i] = 2 * P - PL
        weekly_R2[i] = P + (PH - PL)
    
    # Align weekly pivot levels to daily timeframe
    weekly_P_aligned = align_htf_to_ltf(prices, df_1w, weekly_P)
    weekly_S1_aligned = align_htf_to_ltf(prices, df_1w, weekly_S1)
    weekly_S2_aligned = align_htf_to_ltf(prices, df_1w, weekly_S2)
    weekly_R1_aligned = align_htf_to_ltf(prices, df_1w, weekly_R1)
    weekly_R2_aligned = align_htf_to_ltf(prices, df_1w, weekly_R2)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(weekly_P_aligned[i]) or np.isnan(weekly_S1_aligned[i]) or 
            np.isnan(weekly_S2_aligned[i]) or np.isnan(weekly_R1_aligned[i]) or 
            np.isnan(weekly_R2_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 1w uptrend + volume spike
            long_cond = (close[i] > weekly_R1_aligned[i] and 
                        ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below S1 with 1w downtrend + volume spike
            short_cond = (close[i] < weekly_S1_aligned[i] and 
                         ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and
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