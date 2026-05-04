#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with weekly pivot direction and volume confirmation
# Weekly pivot direction filters breakouts to align with higher timeframe trend, reducing false signals
# Weekly R1/S1 levels act as dynamic support/resistance - breakouts above weekly R1 or below weekly S1 with volume spike
# Uses tighter entry conditions targeting 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
# Weekly trend filter helps in both bull and bear markets by ensuring breakouts align with higher timeframe momentum

name = "6h_Camarilla_R3S3_Breakout_1wPivotDir_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3 = Close + 1.1*(High-Low)/6, S3 = Close - 1.1*(High-Low)/6
    camarilla_range = high_1d - low_1d
    r3 = close_1d + (1.1 * camarilla_range / 6)
    s3 = close_1d - (1.1 * camarilla_range / 6)
    
    # Align to 6h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Weekly trend direction: price above/below weekly pivot
    weekly_uptrend = close_1w > pivot_1w
    weekly_downtrend = close_1w < pivot_1w
    
    # Shift weekly trend to ensure we only use completed weekly bars
    weekly_uptrend_shifted = np.roll(weekly_uptrend, 1)
    weekly_downtrend_shifted = np.roll(weekly_downtrend, 1)
    weekly_uptrend_shifted[0] = False
    weekly_downtrend_shifted[0] = False
    
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend_shifted)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Camarilla R3 AND weekly uptrend AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > r1_1w_aligned[i] and  # Above weekly R1 for stronger bullish bias
                weekly_uptrend_aligned[i] and 
                volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Camarilla S3 AND weekly downtrend AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < s1_1w_aligned[i] and  # Below weekly S1 for stronger bearish bias
                  weekly_downtrend_aligned[i] and 
                  volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla R3 OR below weekly pivot
            if close[i] < r3_aligned[i] or close[i] < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla S3 OR above weekly pivot
            if close[i] > s3_aligned[i] or close[i] > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals