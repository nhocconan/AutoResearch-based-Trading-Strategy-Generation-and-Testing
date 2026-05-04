#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume confirmation
# Uses Camarilla pivot levels (R3/S3) from 4h for breakout structure
# 4h EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (>1.5x 20 EMA volume) filters false breakouts
# Discrete sizing 0.20 targets 60-150 total trades over 4 years (15-37/year)
# Session filter (08-20 UTC) reduces noise trades outside active hours
# Works in bull markets (continuation at R3) and bear markets (continuation at S3)
# Focus on BTC/ETH by requiring 4h trend alignment (avoids SOL-only bias)

name = "1h_Camarilla_R3S3_4hEMA34_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime64 arithmetic in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla calculation and EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA(34) trend filter from prior completed 4h bar
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_shifted = np.roll(ema_34_4h, 1)
    ema_34_4h_shifted[0] = np.nan
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h_shifted)
    
    # Calculate Camarilla levels (R3, S3) from prior completed 4h bar
    # Camarilla: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_range = high_4h - low_4h
    r3_level = close_4h + 1.1 * camarilla_range / 4
    s3_level = close_4h - 1.1 * camarilla_range / 4
    
    # Shift to use prior completed 4h bar
    r3_shifted = np.roll(r3_level, 1)
    s3_shifted = np.roll(s3_level, 1)
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    
    # Align to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND price > 4h EMA34 AND volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_34_4h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3 AND price < 4h EMA34 AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_34_4h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 4h EMA34
            if close[i] < s3_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 4h EMA34
            if close[i] > r3_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals