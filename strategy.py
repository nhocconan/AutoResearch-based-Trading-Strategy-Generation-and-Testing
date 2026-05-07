#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
# Hypothesis: Breakouts at Camarilla R3/S3 levels with 1-day trend filter and volume spike.
# R3/S3 represent stronger support/resistance than R1/S1, reducing false breakouts.
# Uses 1-day EMA50 for trend filter and volume > 2x 20-period average for confirmation.
# Target: 15-25 trades/year to minimize fee decay while capturing strong moves.
# Works in bull/bear by requiring alignment with higher timeframe trend.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels: R3, S3
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 4
    s3 = close_1d - 1.1 * camarilla_range / 4
    
    # Get 1d data for trend filter (EMA50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter on 6h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(ema_50_1d_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: Price > R3, above 1d EMA50 trend, volume spike
            if close[i] > r3_6h[i] and close[i] > ema_50_1d_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Price < S3, below 1d EMA50 trend, volume spike
            elif close[i] < s3_6h[i] and close[i] < ema_50_1d_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: close below R3 or below EMA50
            if close[i] < r3_6h[i] or close[i] < ema_50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close above S3 or above EMA50
            if close[i] > s3_6h[i] or close[i] > ema_50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals