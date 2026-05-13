#!/usr/bin/env python3
"""
6h_12h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: On 6h timeframe, Camarilla R3/S3 breakouts with 1d trend and volume confirmation
provide high-probability continuation trades in both bull and bear markets.
Camarilla levels act as key support/resistance; breakouts above R3 or below S3 with volume
indicate strong momentum in the direction of the daily trend.
"""

name = "6h_12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    R3 = np.zeros(len(df_12h))
    S3 = np.zeros(len(df_12h))
    R4 = np.zeros(len(df_12h))
    S4 = np.zeros(len(df_12h))
    
    for i in range(len(df_12h)):
        H = high_12h[i]
        L = low_12h[i]
        C = close_12h[i]
        R3[i] = C + (H - L) * 1.1 / 6
        S3[i] = C - (H - L) * 1.1 / 6
        R4[i] = C + (H - L) * 1.1 / 2
        S4[i] = C - (H - L) * 1.1 / 2
    
    # Align Camarilla levels to 6h
    R3_6h = align_htf_to_ltf(prices, df_12h, R3)
    S3_6h = align_htf_to_ltf(prices, df_12h, S3)
    R4_6h = align_htf_to_ltf(prices, df_12h, R4)
    S4_6h = align_htf_to_ltf(prices, df_12h, S4)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: 34 EMA
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 6h
    uptrend_1d_6h = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_6h = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume filter: 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if volume filter not met
        if not vol_filter[i]:
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r3 = R3_6h[i]
        s3 = S3_6h[i]
        r4 = R4_6h[i]
        s4 = S4_6h[i]
        uptrend = uptrend_1d_6h[i]
        downtrend = downtrend_1d_6h[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume, in 1d uptrend
            if uptrend and close[i] > r3 and close[i-1] <= r3:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume, in 1d downtrend
            elif downtrend and close[i] < s3 and close[i-1] >= s3:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R4 or 1d trend turns down
            if close[i] >= r4 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S4 or 1d trend turns up
            if close[i] <= s4 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals