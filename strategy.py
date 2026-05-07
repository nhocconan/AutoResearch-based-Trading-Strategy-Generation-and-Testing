#!/usr/bin/env python3
name = "4h_Chaikin_Money_Flow_Trend_200MA_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mts_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 200-period MA on 4h
    ma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Daily Chaikin Money Flow
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Money Flow Multiplier
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d)
    mfm = np.where((high_1d - low_1d) == 0, 0, mfm)
    
    # Money Flow Volume
    mfv = mfm * volume_1d
    
    # 21-period CMF
    mfv_sum = pd.Series(mfv).rolling(window=21, min_periods=21).sum().values
    volume_sum = pd.Series(volume_1d).rolling(window=21, min_periods=21).sum().values
    cmf_21 = np.where(volume_sum != 0, mfv_sum / volume_sum, 0)
    
    cmf_21_aligned = align_htf_to_ltf(prices, df_1d, cmf_21)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(ma_200[i]) or np.isnan(cmf_21_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 200MA and positive CMF
            if close[i] > ma_200[i] and cmf_21_aligned[i] > 0.05:
                signals[i] = 0.30
                position = 1
            # Short: price below 200MA and negative CMF
            elif close[i] < ma_200[i] and cmf_21_aligned[i] < -0.05:
                signals[i] = -0.30
                position = -1
        elif position != 0:
            # Exit: price crosses 200MA or CMF crosses zero
            if position == 1:
                if close[i] < ma_200[i] or cmf_21_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                if close[i] > ma_200[i] or cmf_21_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals