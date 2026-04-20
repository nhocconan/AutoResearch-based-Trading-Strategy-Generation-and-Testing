#!/usr/bin/env python3
# 12h_1d_1w_Pivot_R3S3_Breakout_VolumeTrend
# Hypothesis: On 12h timeframe, trade breakouts from 1d-derived Camarilla R3/S3 levels with volume spike confirmation and 1w EMA50 trend filter.
# Uses 1w EMA50 to filter trades in trending markets. Targets 12-37 trades per year (50-150 total over 4 years).
# Breakouts are confirmed by volume > 2x 20-period average and price beyond 0.5% buffer around R3/S3.
# Exits on reversal below/above S3/R3 or trend flip (price crosses 1w EMA50).

name = "12h_1d_1w_Pivot_R3S3_Breakout_VolumeTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    
    # Pivot point and ranges
    pivot_1d = typical_price_1d
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 and S3
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    
    # 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'])
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d levels to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R3, volume spike, and price above 1w EMA50 (uptrend)
            if (close[i] > r3_aligned[i] * 1.005 and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S3, volume spike, and price below 1w EMA50 (downtrend)
            elif (close[i] < s3_aligned[i] * 0.995 and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S3 or trend reversal (below EMA50)
            if close[i] < s3_aligned[i] * 0.995 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R3 or trend reversal (above EMA50)
            if close[i] > r3_aligned[i] * 1.005 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals