#!/usr/bin/env python3
name = "4H_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels (S1, R1) from previous day
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    range_ = high_1d[:-1] - low_1d[:-1]
    s1 = close_1d[:-1] - 1.05 * range_ / 2.0
    r1 = close_1d[:-1] + 1.05 * range_ / 2.0
    
    # Shift to get previous day's levels
    s1_prev = np.concatenate([[np.nan], s1[:-1]])
    r1_prev = np.concatenate([[np.nan], r1[:-1]])
    
    # Calculate 20-day EMA for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all daily data to 4h timeframe
    s1_prev_aligned = align_htf_to_ltf(prices, df_1d, s1_prev)
    r1_prev_aligned = align_htf_to_ltf(prices, df_1d, r1_prev)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    volume_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    volume_ma[:10] = np.nan  # insufficient data for convolution at start
    volume_confirm = volume > volume_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(s1_prev_aligned[i]) or np.isnan(r1_prev_aligned[i]) or np.isnan(ema20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 + above daily EMA20 + volume confirmation
            if close[i] > r1_prev_aligned[i] and close[i] > ema20_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + below daily EMA20 + volume confirmation
            elif close[i] < s1_prev_aligned[i] and close[i] < ema20_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or below daily EMA20
            if close[i] < s1_prev_aligned[i] or close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or above daily EMA20
            if close[i] > r1_prev_aligned[i] or close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals