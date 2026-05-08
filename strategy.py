#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla pivot levels (R3/S3) with volume confirmation and 1d EMA trend filter.
# Long when price breaks above daily R3 with volume surge and above 1d EMA34.
# Short when price breaks below daily S3 with volume surge and below 1d EMA34.
# Exits when price crosses back below/above daily pivot point.
# Designed for low trade frequency (15-25/year) to avoid fee damp. Camarilla levels from higher timeframe provide structure that works in both trending and ranging markets.

name = "12h_1dCamarilla_R3S3_VolumeTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (R3, S3, pivot) from previous day
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    camarilla_pivot = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(df_1d)):
        # Previous day's range
        range_ = high_1d[i-1] - low_1d[i-1]
        camarilla_pivot[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        camarilla_r3[i] = camarilla_pivot[i] + range_ * 1.1 / 2.0  # R3 = pivot + 1.1*(range/2)
        camarilla_s3[i] = camarilla_pivot[i] - range_ * 1.1 / 2.0  # S3 = pivot - 1.1*(range/2)
    
    # For first day, use same values as second day
    if len(df_1d) >= 2:
        camarilla_pivot[0] = camarilla_pivot[1]
        camarilla_r3[0] = camarilla_r3[1]
        camarilla_s3[0] = camarilla_s3[1]
    elif len(df_1d) == 1:
        camarilla_pivot[0] = close_1d[0]
        camarilla_r3[0] = close_1d[0]
        camarilla_s3[0] = close_1d[0]
    
    # Calculate 1d EMA34
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 12h volume spike (2x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above daily R3 + volume surge + above 1d EMA34
            if close[i] > camarilla_r3_aligned[i] and vol_spike[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below daily S3 + volume surge + below 1d EMA34
            elif close[i] < camarilla_s3_aligned[i] and vol_spike[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below daily pivot point
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above daily pivot point
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals