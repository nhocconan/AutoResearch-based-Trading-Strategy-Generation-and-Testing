#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_12hTrend_Volume_Adjusted
# Hypothesis: Uses Camarilla R3/S3 levels from 1d as breakout levels, filtered by 12h EMA50 trend and volume spikes.
# Adjustments: Reduced trade frequency by requiring stronger volume spikes (3x avg) and adding a minimum hold period of 4 bars to avoid whipsaws.
# Designed for 4h timeframe to balance trade frequency and signal quality. Works in bull/bear via trend filter.

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume_Adjusted"
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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume spike on 4h timeframe (20-period average, 3x threshold for strength)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (3.0 * vol_ma_20)  # Increased from 2.0 to 3.0 to reduce trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track bars since entry to enforce minimum hold
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or 
            np.isnan(ema_50_12h_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 + above 12h EMA50 + volume spike
            if close[i] > camarilla_r3_4h[i] and close[i] > ema_50_12h_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Price breaks below Camarilla S3 + below 12h EMA50 + volume spike
            elif close[i] < camarilla_s3_4h[i] and close[i] < ema_50_12h_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            bars_since_entry += 1
            # Exit: Price closes below Camarilla S3 or below 12h EMA50, but only after minimum 4 bars
            if bars_since_entry >= 4 and (close[i] < camarilla_s3_4h[i] or close[i] < ema_50_12h_4h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            bars_since_entry += 1
            # Exit: Price closes above Camarilla R3 or above 12h EMA50, but only after minimum 4 bars
            if bars_since_entry >= 4 and (close[i] > camarilla_r3_4h[i] or close[i] > ema_50_12h_4h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals