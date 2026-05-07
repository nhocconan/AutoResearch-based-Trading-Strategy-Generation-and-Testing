#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34) and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, S3) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d_prev) / 3.0
    r3 = pivot + (high_1d - low_1d) * 1.1 / 2.0
    s3 = pivot - (high_1d - low_1d) * 1.1 / 2.0
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d volume average for volume spike filter
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 (resistance) in uptrend with volume spike
            if (high[i] > r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > vol_avg_20_aligned[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 (support) in downtrend with volume spike
            elif (low[i] < s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > vol_avg_20_aligned[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA34 (trend change) or breaks below S3 (reversal)
            if close[i] < ema_34_1d_aligned[i] or low[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA34 (trend change) or breaks above R3 (reversal)
            if close[i] > ema_34_1d_aligned[i] or high[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals