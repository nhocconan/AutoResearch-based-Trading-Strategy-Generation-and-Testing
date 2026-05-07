#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_12hTrend_Volume_Spike_v1"
timeframe = "4h"
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
    
    # Load 4h and 12h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_4h) < 20 or len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h Close for trend direction
    close_12h = df_12h['close'].values
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # 1d Camarilla levels: R3, S3 from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    camarilla_r3_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 2
    camarilla_s3_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 2
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # 4h volume spike: > 2.5x 20-period average
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = volume > 2.5 * vol_ma_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(close_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with volume spike, uptrend (12h close > EMA50)
            if (close[i] > camarilla_r3_1d_aligned[i] and vol_spike_4h[i] and 
                close_12h_aligned[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume spike, downtrend (12h close < EMA50)
            elif (close[i] < camarilla_s3_1d_aligned[i] and vol_spike_4h[i] and 
                  close_12h_aligned[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below S3 or trend reversal (12h close < EMA50)
            if close[i] < camarilla_s3_1d_aligned[i] or close_12h_aligned[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above R3 or trend reversal (12h close > EMA50)
            if close[i] > camarilla_r3_1d_aligned[i] or close_12h_aligned[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals