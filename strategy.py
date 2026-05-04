#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume-weighted price trend filter
# Uses Camarilla pivot levels (R3/S3) from 12h for proven breakout structure
# 1d VWAP trend ensures alignment with institutional money flow direction
# Volume confirmation (>1.5x 20 EMA volume) filters false breakouts
# Discrete sizing 0.25 targets 50-150 trades over 4 years (12-37/year)
# Works in bull markets (continuation at R3) and bear markets (continuation at S3)
# BTC/ETH focus: requires 1d VWAP alignment to avoid SOL-only bias

name = "12h_Camarilla_R3S3_1dVWAPTrend_VolumeConfirm_Balanced"
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
    
    # Get 1d data for VWAP trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for VWAP calculation
        return np.zeros(n)
    
    # Calculate 1d VWAP from prior completed 1d bar
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vwap_num = (typical_price_1d * df_1d['volume'].values).cumsum()
    vwap_den = df_1d['volume'].values.cumsum()
    vwap_1d = vwap_num / vwap_den
    vwap_1d_shifted = np.roll(vwap_1d, 1)
    vwap_1d_shifted[0] = np.nan
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_shifted)
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Camarilla levels (R3, S3) from prior completed 12h bar
    camarilla_range = high_12h - low_12h
    r3_level = close_12h + 1.1 * camarilla_range / 2
    s3_level = close_12h - 1.1 * camarilla_range / 2
    
    # Shift to use prior completed 12h bar
    r3_shifted = np.roll(r3_level, 1)
    s3_shifted = np.roll(s3_level, 1)
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND price > 1d VWAP AND volume spike
            if close[i] > r3_aligned[i] and close[i] > vwap_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND price < 1d VWAP AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < vwap_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 1d VWAP
            if close[i] < s3_aligned[i] or close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 1d VWAP
            if close[i] > r3_aligned[i] or close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals