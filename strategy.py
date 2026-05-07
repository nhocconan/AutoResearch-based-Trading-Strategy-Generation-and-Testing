#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Uses Camarilla R3/S3 levels from daily timeframe with 1d EMA34 trend filter and volume confirmation.
# In bull markets, price above EMA34 with breakout above R3 triggers long; in bear markets, price below EMA34 with breakdown below S3 triggers short.
# Volume spikes confirm institutional participation. Designed for 12h to balance trade frequency and signal quality.
# Works in both bull and bear via trend-following Camarilla breakout signals.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # Camarilla: R3 = close + (high - low) * 1.1/2, S3 = close - (high - low) * 1.1/2
    camarilla_range = high_1d - low_1d
    r3 = close_1d + camarilla_range * 1.1 / 2
    s3 = close_1d - camarilla_range * 1.1 / 2
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike on 12h timeframe (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema_34_1d_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + above 1d EMA34 trend + volume spike
            if close[i] > r3_12h[i] and close[i] > ema_34_1d_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + below 1d EMA34 trend + volume spike
            elif close[i] < s3_12h[i] and close[i] < ema_34_1d_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls below EMA34 or loses bullish momentum
            if close[i] < ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises above EMA34 or loses bearish momentum
            if close[i] > ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals