#!/usr/bin/env python3
# 4h_12hCAMARILLA_R1S1_BREAKOUT_12HTREND_VOLUME
# Breakout above CAMARILLA_R1S1 (1d) with 12h EMA50 trend and volume confirmation.
# Long when price breaks above R1S1 in uptrend, short when breaks below S1S1 in downtrend.
# Uses volume spike and minimum holding period to reduce churn. Designed for 4h timeframe
# to work in both bull and bear markets via trend filter and volatility-based entries.

name = "4h_12hCAMARILLA_R1S1_BREAKOUT_12HTREND_VOLUME"
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
    
    # Get daily data for CAMARILLA levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate CAMARILLA levels from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's data (no look-ahead)
    R1 = np.roll(close_1d + (high_1d - low_1d) * 1.1 / 12, 1)
    S1 = np.roll(close_1d - (high_1d - low_1d) * 1.1 / 12, 1)
    R1[0] = np.nan
    S1[0] = np.nan
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    ema_50_12h_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: 20-period MA on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(ema_50_12h_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above R1 with uptrend and volume
            if close[i] > R1_4h[i] and close[i] > ema_50_12h_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below S1 with downtrend and volume
            elif close[i] < S1_4h[i] and close[i] < ema_50_12h_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price returns below EMA50 or breaks below S1
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry >= 3 and (close[i] < ema_50_12h_4h[i] or close[i] < S1_4h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above EMA50 or breaks above R1
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry >= 3 and (close[i] > ema_50_12h_4h[i] or close[i] > R1_4h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals