#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above R3 with 1d uptrend and volume > 2x average.
# Short when price breaks below S3 with 1d downtrend and volume > 2x average.
# Exit on opposite break or volume normalization.
# Uses 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in bull (breakout continuation) and bear (breakdown continuation).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    camarilla_range = high_12h - low_12h
    r3_level = close_12h + 1.1 * camarilla_range
    s3_level = close_12h - 1.1 * camarilla_range
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = ema_34_1d > np.roll(ema_34_1d, 1)
    trend_1d_up = np.where(np.isnan(trend_1d_up), False, trend_1d_up)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Align 12h Camarilla levels to 12h index (no additional delay needed for breakout)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_level)
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, 1d trend up, volume spike
            if (close[i] > r3_aligned[i] and trend_1d_up_aligned[i] and vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, 1d trend down, volume spike
            elif (close[i] < s3_aligned[i] and not trend_1d_up_aligned[i] and vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or volume normalizes
            if (close[i] < s3_aligned[i] or vol_ratio[i] < 1.1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 or volume normalizes
            if (close[i] > r3_aligned[i] or vol_ratio[i] < 1.1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals