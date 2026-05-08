#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla pivot points (R3/S3) with volume confirmation and 1w EMA trend filter.
# Long when price breaks above daily R3 with volume surge and above 1w EMA.
# Short when price breaks below daily S3 with volume surge and below 1w EMA.
# Exits when price returns to daily pivot point. Designed for low trade frequency (12-37/year) to avoid fee drag.
# Camarilla levels provide strong support/resistance that works in both trending and ranging markets.

name = "12h_Camarilla_R3_S3_VolumeTrend"
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
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (R3, S3, pivot)
    camarilla_pivot = np.full_like(close_1d, np.nan)
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        # Camarilla formulas using previous day's OHLC
        if i == 0:
            # For first day, use same day's OHLC
            camarilla_pivot[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        else:
            camarilla_pivot[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        
        camarilla_r3[i] = camarilla_pivot[i] + (high_1d[i-1] - low_1d[i-1]) * 1.1 / 4
        camarilla_s3[i] = camarilla_pivot[i] - (high_1d[i-1] - low_1d[i-1]) * 1.1 / 4
    
    # Calculate 1w EMA (using 1d data as proxy - 5 days ~ 1 week)
    ema_1w = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Align 1d indicators to 12h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1d, ema_1w)
    
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
            np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above daily R3 + volume surge + above 1w EMA
            if close[i] > camarilla_r3_aligned[i] and vol_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below daily S3 + volume surge + below 1w EMA
            elif close[i] < camarilla_s3_aligned[i] and vol_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to daily pivot
            if close[i] <= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to daily pivot
            if close[i] >= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals