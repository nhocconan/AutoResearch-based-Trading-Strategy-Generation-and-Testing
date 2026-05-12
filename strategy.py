#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d
    def calculate_camarilla(high_val, low_val, close_val):
        range_val = high_val - low_val
        if range_val <= 0:
            return close_val, close_val, close_val, close_val, close_val, close_val, close_val, close_val
        multiplier = range_val / 12.0
        S1 = close_val - multiplier * 1.1
        S2 = close_val - multiplier * 2.0
        S3 = close_val - multiplier * 3.0
        R3 = close_val + multiplier * 3.0
        R2 = close_val + multiplier * 2.0
        R1 = close_val + multiplier * 1.1
        return S1, S2, S3, R3, R2, R1
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    S1_1d = np.full_like(close_1d, np.nan)
    S2_1d = np.full_like(close_1d, np.nan)
    S3_1d = np.full_like(close_1d, np.nan)
    R3_1d = np.full_like(close_1d, np.nan)
    R2_1d = np.full_like(close_1d, np.nan)
    R1_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        S1, S2, S3, R3, R2, R1 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        S1_1d[i] = S1
        S2_1d[i] = S2
        S3_1d[i] = S3
        R3_1d[i] = R3
        R2_1d[i] = R2
        R1_1d[i] = R1
    
    # Align Camarilla levels to 12h timeframe
    S1_12h = align_htf_to_ltf(prices, df_1d, S1_1d)
    S2_12h = align_htf_to_ltf(prices, df_1d, S2_1d)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3_1d)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3_1d)
    R2_12h = align_htf_to_ltf(prices, df_1d, R2_1d)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1_1d)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough data for 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R3 with volume spike + 1d uptrend
            if (close[i] > R3_12h[i] and 
                volume_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 with volume spike + 1d downtrend
            elif (close[i] < S3_12h[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when close crosses below R2 or 1d trend turns down
            if (close[i] < R2_12h[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when close crosses above S2 or 1d trend turns up
            if (close[i] > S2_12h[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals