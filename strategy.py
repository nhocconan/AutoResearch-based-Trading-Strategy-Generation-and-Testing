#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Camarilla pivots identify key intraday support/resistance levels. Breaks of R3/S3 with
# volume spike and 1w EMA34 trend filter capture strong momentum moves. Designed for 1d timeframe
# to target 30-100 trades over 4 years (7-25/year) minimizing fee drag. Works in bull markets via
# breakout continuation and in bear markets via mean reversion from extreme levels.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeConfirmation"
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
    
    # Get 1d data for Camarilla pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d (based on previous day)
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1/2
    # S3 = C - (H - L) * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.1 / 2.0
    s3_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 1d timeframe (wait for completed 1d bar)
    # Since we're already on 1d timeframe, we need to shift by 1 to avoid look-ahead
    r3_1d_aligned = np.roll(r3_1d, 1)
    s3_1d_aligned = np.roll(s3_1d, 1)
    # Set first value to NaN as there's no previous day
    r3_1d_aligned[0] = np.nan
    s3_1d_aligned[0] = np.nan
    
    # Get 1w data for EMA34 trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 1d timeframe (wait for completed 1w bar)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Close crosses above R3 AND volume spike AND 1w EMA34 uptrend
            if (close[i] > r3_1d_aligned[i] and 
                close[i-1] <= r3_1d_aligned[i-1] and  # crossed above R3 from below
                volume_spike[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Close crosses below S3 AND volume spike AND 1w EMA34 downtrend
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i-1] >= s3_1d_aligned[i-1] and  # crossed below S3 from above
                  volume_spike[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close crosses below R3 (breakdown) OR trend reverses
            if close[i] < r3_1d_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close crosses above S3 (breakout) OR trend reverses
            if close[i] > s3_1d_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals