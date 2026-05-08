#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot breakout with daily trend filter and volume confirmation
# We go long when price breaks above the daily Camarilla R3 level with daily EMA(34) uptrend and volume spike.
# We go short when price breaks below the daily Camarilla S3 level with daily EMA(34) downtrend and volume spike.
# The 12h timeframe reduces trade frequency while Camarilla levels provide institutional support/resistance.
# Daily trend filter ensures alignment with higher timeframe momentum. Volume spike confirms breakout strength.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) with proper risk control.

name = "12h_Camarilla_R3S3_Breakout_DailyTrend_Volume"
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
    
    # Get daily data once for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Pivot point calculation
    pivot = (high_1d + low_1d + close_1d_prev) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/2, S3 = close - (high - low) * 1.1/2
    camarilla_r3 = close_1d_prev + (range_1d * 1.1 / 2)
    camarilla_s3 = close_1d_prev - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: current volume > 2.0 * 20-period average (on 12h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Camarilla R3 + daily uptrend + volume spike
            if (close[i] > r3_level and 
                close[i] > ema34_1d_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S3 + daily downtrend + volume spike
            elif (close[i] < s3_level and 
                  close[i] < ema34_1d_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 OR daily trend turns down
            if (close[i] < s3_level or close[i] < ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 OR daily trend turns up
            if (close[i] > r3_level or close[i] > ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals