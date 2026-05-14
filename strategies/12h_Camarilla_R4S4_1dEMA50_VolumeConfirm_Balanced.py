#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1d trend filter and volume confirmation
# Uses Camarilla pivot levels (R4/S4) from 12h for stronger breakout structure than R3/S3
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (>1.4x 20 EMA volume) filters false breakouts
# Discrete sizing 0.25 minimizes fee churn while targeting 50-150 trades over 4 years
# Works in bull markets (continuation at R4) and bear markets (continuation at S4)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "12h_Camarilla_R4S4_1dEMA50_VolumeConfirm_Balanced"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_shifted = np.roll(ema_50_1d, 1)
    ema_50_1d_shifted[0] = np.nan
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_shifted)
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Camarilla levels (R4, S4) from prior completed 12h bar
    camarilla_range = high_12h - low_12h
    r4_level = close_12h + 1.1 * camarilla_range
    s4_level = close_12h - 1.1 * camarilla_range
    
    # Shift to use prior completed 12h bar
    r4_shifted = np.roll(r4_level, 1)
    s4_shifted = np.roll(s4_level, 1)
    r4_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    # Align to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4_shifted)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R4 AND price > 1d EMA50 AND volume spike
            if close[i] > r4_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > (1.4 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S4 AND price < 1d EMA50 AND volume spike
            elif close[i] < s4_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > (1.4 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S4 OR price crosses below 1d EMA50
            if close[i] < s4_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R4 OR price crosses above 1d EMA50
            if close[i] > r4_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals