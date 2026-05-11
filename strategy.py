#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Daily data for Camarilla pivots, EMA, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Camarilla levels from previous day
    close_prev = np.roll(close_1d, 1)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev[0] = np.nan
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    
    # Calculate R3 and S3 levels
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # Daily EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily volume spike (20-day average)
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma * 2.0)
    
    # Align all 1d data to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # Wait for EMA34 to be valid
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(ema34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + uptrend + volume spike
            if close[i] > R3_aligned[i] and close[i] > ema34_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + downtrend + volume spike
            elif close[i] < S3_aligned[i] and close[i] < ema34_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA34
            if close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA34
            if close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals