#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1D HTF for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels (R3, S3) from previous day
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    range_ = high_1d[:-1] - low_1d[:-1]
    r3 = close_1d[:-1] + range_ * 1.1 / 2
    s3 = close_1d[:-1] - range_ * 1.1 / 2
    
    # Shift to align with current day (use previous day's levels)
    r3 = np.concatenate([[np.nan], r3[:-1]])
    s3 = np.concatenate([[np.nan], s3[:-1]])
    
    # 1D EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # Align 1D data to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R3 + uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_aligned[i] and vol_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: Close breaks below S3 + downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_aligned[i] and vol_spike[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Close breaks below S3 or trend reversal
            if close[i] < s3_aligned[i] or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Close breaks above R3 or trend reversal
            if close[i] > r3_aligned[i] or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals