#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from 1d for structure, 1w EMA50 for trend filter (proven from top performers),
# and volume spike for confirmation. Designed for 10-25 trades/year to minimize fee drag.
# Works in bull markets via breakout continuations and in bear markets via breakdown continuations.
# The 1w EMA50 provides a smooth trend filter that adapts to changing regimes while avoiding whipsaw.

name = "1d_Camarilla_R3S3_1wEMA50_VolumeSpike_TrendFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter from prior completed 1w bar
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_shifted = np.roll(ema50_1w, 1)
    ema50_1w_shifted[0] = np.nan
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_shifted)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from prior completed 1d bar
    # Camarilla: R3 = close + 1.125*(high-low), S3 = close - 1.125*(high-low)
    R3_1d = close_1d + 1.125 * (high_1d - low_1d)
    S3_1d = close_1d - 1.125 * (high_1d - low_1d)
    R3_1d_shifted = np.roll(R3_1d, 1)
    S3_1d_shifted = np.roll(S3_1d, 1)
    R3_1d_shifted[0] = np.nan
    S3_1d_shifted[0] = np.nan
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d_shifted)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(R3_1d_aligned[i]) or
            np.isnan(S3_1d_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND above 1w EMA50 AND volume spike
            if close[i] > R3_1d_aligned[i] and close[i] > ema50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND below 1w EMA50 AND volume spike
            elif close[i] < S3_1d_aligned[i] and close[i] < ema50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 OR below 1w EMA50
            if close[i] < S3_1d_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R3 OR above 1w EMA50
            if close[i] > R3_1d_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals