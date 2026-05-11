#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_S3 = np.zeros(len(close_1d))
    camarilla_R3 = np.zeros(len(close_1d))
    camarilla_S1 = np.zeros(len(close_1d))
    camarilla_R1 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_S3[i] = camarilla_R3[i] = camarilla_S1[i] = camarilla_R1[i] = close_1d[i]
            continue
        range_ = high_1d[i-1] - low_1d[i-1]
        camarilla_S3[i] = close_1d[i-1] - (range_ * 1.1 / 4)
        camarilla_R3[i] = close_1d[i-1] + (range_ * 1.1 / 4)
        camarilla_S1[i] = close_1d[i-1] - (range_ * 1.1 / 6)
        camarilla_R1[i] = close_1d[i-1] + (range_ * 1.1 / 6)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Require strong volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # EMA50 and volume MA
    
    for i in range(start_idx, n):
        if np.isclose(close[i], 0) or np.isnan(ema50_1w_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and weekly uptrend
            if close[i] > camarilla_R3_aligned[i] and vol_spike[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and weekly downtrend
            elif close[i] < camarilla_S3_aligned[i] and vol_spike[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 (reversal signal)
            if close[i] < camarilla_S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 (reversal signal)
            if close[i] > camarilla_R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals