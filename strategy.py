#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h volume spike and 1d EMA trend filter
# Camarilla levels identify high-probability reversal/continuation zones. 
# Breakout above R3 or below S3 with volume surge indicates institutional breakout.
# 1d EMA50 filter ensures we only trade in direction of higher timeframe trend.
# Targets 20-30 trades per year (~80-120 total over 4 years) to minimize fee drag.
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered by EMA).

name = "4h_Camarilla_R3S3_12hVolume_1dEMA"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 4h (using previous bar's range)
    camarilla_r3 = np.full_like(high, np.nan)
    camarilla_s3 = np.full_like(low, np.nan)
    
    for i in range(1, n):
        # Use previous bar's high-low range
        prev_range = high[i-1] - low[i-1]
        if prev_range <= 0:
            camarilla_r3[i] = camarilla_r3[i-1]  # carry forward last valid
            camarilla_s3[i] = camarilla_s3[i-1]
        else:
            close_prev = close[i-1]
            camarilla_r3[i] = close_prev + 1.1 * prev_range / 6.0
            camarilla_s3[i] = close_prev - 1.1 * prev_range / 6.0
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = vol_12h > (vol_ma_12h * 2.0)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, 50)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_spike_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3, volume spike, price above 1d EMA50 (uptrend)
            if close[i] > camarilla_r3[i] and vol_spike_12h_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, volume spike, price below 1d EMA50 (downtrend)
            elif close[i] < camarilla_s3[i] and vol_spike_12h_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 level or breaks below EMA50
            if close[i] < camarilla_s3[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 level or breaks above EMA50
            if close[i] > camarilla_r3[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals