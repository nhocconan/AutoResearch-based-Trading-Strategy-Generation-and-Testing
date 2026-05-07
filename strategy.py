#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: 6h chart strategy using Camarilla R3/S3 breakouts with 12h EMA50 trend filter and volume confirmation.
# R3/S3 levels represent stronger support/resistance than R1/S1, reducing false breakouts.
# Trend filter ensures trading with higher timeframe momentum. Volume confirms breakout strength.
# Designed to work in both bull and bear markets by capturing strong directional moves with proper filtering.
# Target: 15-30 trades/year per symbol to minimize fee drag while maintaining edge.

timeframe = "6h"
name = "6h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h closes for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    camarilla_r3 = d_close + 1.1 * (d_high - d_low) * 2 / 4  # R3 = C + 1.1*(H-L)*2/4
    camarilla_s3 = d_close - 1.1 * (d_high - d_low) * 2 / 4  # S3 = C - 1.1*(H-L)*2/4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike detection: 2.0x average volume (4-period = 1 day on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 4)  # Ensure we have EMA and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close > R3 with volume spike and price above 12h EMA50
            if (close[i] > camarilla_r3_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: close < S3 with volume spike and price below 12h EMA50
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: touch S3 (opposite level) or trend failure (price below 12h EMA50)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: touch R3 (opposite level) or trend failure (price above 12h EMA50)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals