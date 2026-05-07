#!/usr/bin/env python3
name = "12h_1d_Camarilla_R3S3_Breakout_Trend_Volume"
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
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d Previous day's close for trend filter
    prev_close_1d = df_1d['close'].shift(1).values
    # 12h EMA50 for trend confirmation
    ema50_12h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d Camarilla levels: R3, S3 from previous day
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    camarilla_r3_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 2
    camarilla_s3_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 2
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # 12h volume spike: > 2.0x 30-period average
    vol_ma_12h = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike_12h = volume > 2.0 * vol_ma_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or 
            np.isnan(prev_close_1d[i]) or np.isnan(ema50_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with volume spike, bullish trend (close > EMA50)
            if (close[i] > camarilla_r3_1d_aligned[i] and vol_spike_12h[i] and 
                close[i] > ema50_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume spike, bearish trend (close < EMA50)
            elif (close[i] < camarilla_s3_1d_aligned[i] and vol_spike_12h[i] and 
                  close[i] < ema50_12h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below S3 or trend turns bearish (close < EMA50)
            if close[i] < camarilla_s3_1d_aligned[i] or close[i] < ema50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above R3 or trend turns bullish (close > EMA50)
            if close[i] > camarilla_r3_1d_aligned[i] or close[i] > ema50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals