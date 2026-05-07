#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: Uses daily Camarilla pivot levels (R3/S3) for breakout entries with 12h EMA trend filter and volume spike confirmation.
# The Camarilla R3/S3 levels represent strong support/resistance that often trigger breakouts in both bull and bear markets.
# The 12h EMA filter ensures alignment with intermediate trend, while volume confirms breakout strength.
# Designed for low trade frequency (<50/year) to minimize fee drag and maximize edge persistence.

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
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
    
    # Get daily data for Camarilla pivot points (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # R3 = H + 2*(H-L)*1.1/2 = H + 1.1*(H-L)
    # S3 = L - 2*(H-L)*1.1/2 = L - 1.1*(H-L)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r3 = high_1d + 1.1 * (high_1d - low_1d)
    s3 = low_1d - 1.1 * (high_1d - low_1d)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate 12h EMA20 for trend filter
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    ema_20_12h_4h = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume spike filter on 4h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema_20_12h_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: Price > daily R3, above 12h EMA20 trend, volume spike
            if close[i] > r3_4h[i] and close[i] > ema_20_12h_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Price < daily S3, below 12h EMA20 trend, volume spike
            elif close[i] < s3_4h[i] and close[i] < ema_20_12h_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price breaks below daily R3 or below 12h EMA20
            if close[i] < r3_4h[i] or close[i] < ema_20_12h_4h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above daily S3 or above 12h EMA20
            if close[i] > s3_4h[i] or close[i] > ema_20_12h_4h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals