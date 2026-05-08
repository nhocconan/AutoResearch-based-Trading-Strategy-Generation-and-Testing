#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous day's data
    # H = high, L = low, C = close of previous day
    n1d = len(close_1d)
    H1 = high_1d[0]  # dummy for index 0
    L1 = low_1d[0]
    C1 = close_1d[0]
    
    S1 = np.full(n1d, np.nan)
    S2 = np.full(n1d, np.nan)
    S3 = np.full(n1d, np.nan)
    S4 = np.full(n1d, np.nan)
    R1 = np.full(n1d, np.nan)
    R2 = np.full(n1d, np.nan)
    R3 = np.full(n1d, np.nan)
    R4 = np.full(n1d, np.nan)
    
    for i in range(1, n1d):
        H = high_1d[i-1]  # Previous day high
        L = low_1d[i-1]   # Previous day low
        C = close_1d[i-1] # Previous day close
        
        R4 = C + (H - L) * 1.1 / 2
        R3 = C + (H - L) * 1.1 / 4
        R2 = C + (H - L) * 1.1 / 6
        R1 = C + (H - L) * 1.1 / 12
        S1 = C - (H - L) * 1.1 / 12
        S2 = C - (H - L) * 1.1 / 6
        S3 = C - (H - L) * 1.1 / 4
        S4 = C - (H - L) * 1.1 / 2
        
        R1_arr[i] = R1
        R2_arr[i] = R2
        R3_arr[i] = R3
        R4_arr[i] = R4
        S1_arr[i] = S1
        S2_arr[i] = S2
        S3_arr[i] = S3
        S4_arr[i] = S4
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1_arr)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2_arr)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3_arr)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4_arr)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1_arr)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2_arr)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3_arr)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4_arr)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 1d uptrend + volume spike
            long_cond = (close[i] > R1_aligned[i] and 
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below S1 with 1d downtrend + volume spike
            short_cond = (close[i] < S1_aligned[i] and 
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals