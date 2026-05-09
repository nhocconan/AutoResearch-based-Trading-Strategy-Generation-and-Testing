#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume spike
# Uses Camarilla pivot levels from 1d, filtered by 1d EMA(34) trend and volume spike (>1.5x avg volume)
# Only takes longs when price > 1d Camarilla R3 AND 1d EMA(34) rising AND volume spike
# Only takes shorts when price < 1d Camarilla S3 AND 1d EMA(34) falling AND volume spike
# Exits when price crosses the 1d Camarilla midpoint (P) or trend reverses
# Target: 20-50 trades per year with position size 0.25 (tight entry to avoid overtrading)

name = "4h_Camarilla_R3S3_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close']
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_prev = np.roll(ema_34_1d, 1)
    ema_34_1d_prev[0] = ema_34_1d[0]
    ema_rising = ema_34_1d > ema_34_1d_prev
    ema_falling = ema_34_1d < ema_34_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Calculate Camarilla pivot levels from 1d
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d_series = df_1d['close']
    pivot = (high_1d + low_1d + close_1d_series) / 3
    range_hl = high_1d - low_1d
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    # Camarilla midpoint is the pivot point (P)
    mid = pivot
    
    r3_values = r3.values
    s3_values = s3.values
    mid_values = mid.values
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_values)
    mid_aligned = align_htf_to_ltf(prices, df_1d, mid_values)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(mid_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > 1d Camarilla R3 + 1d EMA rising + volume spike
            if (close[i] > r3_aligned[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < 1d Camarilla S3 + 1d EMA falling + volume spike
            elif (close[i] < s3_aligned[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1d Camarilla midpoint OR trend turns down
            if (close[i] < mid_aligned[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 1d Camarilla midpoint OR trend turns up
            if (close[i] > mid_aligned[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals