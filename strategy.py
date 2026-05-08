#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter: EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_low_range = high_1d - low_1d
    camarilla_high = high_1d + 1.1 * high_low_range
    camarilla_low = low_1d - 1.1 * high_low_range
    camarilla_range = camarilla_high - camarilla_low
    R3 = camarilla_low + camarilla_range * 1.1000
    S3 = camarilla_high - camarilla_range * 1.1000
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above R3, above daily EMA34, volume confirmation
            if close[i] > R3_4h[i] and close[i] > ema_34_1d_aligned[i] and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Price below S3, below daily EMA34, volume confirmation
            elif close[i] < S3_4h[i] and close[i] < ema_34_1d_aligned[i] and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price below S3 or below daily EMA34
            if close[i] < S3_4h[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price above R3 or above daily EMA34
            if close[i] > R3_4h[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals