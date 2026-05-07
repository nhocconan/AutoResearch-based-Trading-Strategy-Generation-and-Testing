#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Combines 4h price action with 12h trend filter to reduce false breakouts.
# Uses Camarilla R1/S1 from 1d for entry levels, filtered by 12h EMA50 trend and 4h volume spikes.
# In bull markets: 12h trend up + price breaks R1 with volume = long continuation.
# In bear markets: 12h trend down + price breaks S1 with volume = short continuation.
# The 12h trend filter reduces whipsaws vs 1d, improving robustness in both regimes.
# Target: 20-40 trades/year to minimize fee drag while maintaining edge.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels: R1, S1
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    ema_50_12h_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike filter on 4h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_50_12h_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > R1, above 12h EMA50 trend, volume spike
            if close[i] > r1_4h[i] and close[i] > ema_50_12h_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < S1, below 12h EMA50 trend, volume spike
            elif close[i] < s1_4h[i] and close[i] < ema_50_12h_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < R1 or below 12h EMA50 trend
            if close[i] < r1_4h[i] or close[i] < ema_50_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > S1 or above 12h EMA50 trend
            if close[i] > s1_4h[i] or close[i] > ema_50_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals