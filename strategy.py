#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels from weekly data
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R1, R2, R3, S1, S2, S3
    R1 = pivot + (range_1w * 1.0833 / 12)
    R2 = pivot + (range_1w * 1.0833 / 6)
    R3 = pivot + (range_1w * 1.0833 / 4)
    S1 = pivot - (range_1w * 1.0833 / 12)
    S2 = pivot - (range_1w * 1.0833 / 6)
    S3 = pivot - (range_1w * 1.0833 / 4)
    
    # Align Camarilla levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Weekly trend filter: EMA(21)
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(ema_21_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R1 + weekly uptrend + volume confirmation
            if (close[i] > R1_aligned[i] and
                close[i] > ema_21_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 + weekly downtrend + volume confirmation
            elif (close[i] < S1_aligned[i] and
                  close[i] < ema_21_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below pivot or weekly downtrend
            if (close[i] < pivot_aligned[i] or
                close[i] < ema_21_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above pivot or weekly uptrend
            if (close[i] > pivot_aligned[i] or
                close[i] > ema_21_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals