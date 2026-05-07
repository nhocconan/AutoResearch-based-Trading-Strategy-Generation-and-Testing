#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: 4h chart strategy using Camarilla pivot breakouts filtered by 12h trend (EMA50) and volume confirmation (1.5x average). 
# Camarilla R3/S3 levels act as support/resistance with high probability of reversal or breakout. 
# 12h EMA50 provides trend filter to avoid counter-trend trades. Volume confirms breakout validity.
# Target: 20-40 trades/year per symbol to minimize fee drag while maintaining edge in both bull and bear markets.

timeframe = "4h"
name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
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
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate daily data for Camarilla pivot points (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla pivot points: R3, S3
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike detection: 1.5x average volume (6-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 6)  # Ensure we have EMA50 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with volume, and 12h trend is bullish (price > EMA50)
            if (high[i] > camarilla_r3_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 with volume, and 12h trend is bearish (price < EMA50)
            elif (low[i] < camarilla_s3_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Camarilla S3 (reversal signal)
            if low[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Camarilla R3 (reversal signal)
            if high[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals