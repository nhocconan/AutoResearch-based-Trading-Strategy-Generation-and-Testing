#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 8:
        return np.zeros(n)
    
    # Calculate weekly EMA(8) for trend direction
    close_1w = df_1w['close'].values
    ema8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema8_1w)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    # Set first value to NaN as there is no previous day
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    close_1d_shift[0] = np.nan
    
    # Calculate pivot and Camarilla levels using previous day's data
    pivot = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    range_ = high_1d_shift - low_1d_shift
    r3 = close_1d_shift + (range_ * 1.1 / 4)
    s3 = close_1d_shift - (range_ * 1.1 / 4)
    
    # Align Camarilla levels to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema8_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema8_1w_val = ema8_1w_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above S3 + uptrend + volume spike
            if (close[i] > s3_val and 
                close[i] > ema8_1w_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below R3 + downtrend + volume spike
            elif (close[i] < r3_val and 
                  close[i] < ema8_1w_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR trend turns down
            if (close[i] < s3_val or close[i] < ema8_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR trend turns up
            if (close[i] > r3_val or close[i] > ema8_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals