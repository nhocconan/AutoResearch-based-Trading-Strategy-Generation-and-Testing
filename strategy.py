#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3/S3) from 1d for balanced breakout structure
# 1w EMA34 ensures alignment with weekly trend to avoid counter-trend whipsaws
# Volume confirmation (>1.5x 50 EMA volume) filters false breakouts
# Discrete sizing 0.25 targets 50-150 trades over 4 years (~12-37/year)
# Works in bull markets (continuation at R3) and bear markets (continuation at S3)
# Focus on BTC/ETH by requiring 1w trend alignment (avoids SOL-only bias)

name = "12h_Camarilla_R3S3_1wEMA34_VolumeConfirm_Balanced"
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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for Camarilla
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34) trend filter from prior completed 1w bar
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_shifted = np.roll(ema_34_1w, 1)
    ema_34_1w_shifted[0] = np.nan
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_shifted)
    
    # Calculate Camarilla levels (R3, S3) from prior completed 1d bar
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_range = high_1d - low_1d
    r3_level = close_1d + (1.1 * camarilla_range) / 2
    s3_level = close_1d - (1.1 * camarilla_range) / 2
    
    # Shift to use prior completed 1d bar
    r3_shifted = np.roll(r3_level, 1)
    s3_shifted = np.roll(s3_level, 1)
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_shifted)
    
    # Volume confirmation: 50-period EMA of volume
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND price > 1w EMA34 AND volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume[i] > (1.5 * vol_ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND price < 1w EMA34 AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume[i] > (1.5 * vol_ema_50[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 1w EMA34
            if close[i] < s3_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 1w EMA34
            if close[i] > r3_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals