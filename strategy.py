#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R4_S4_Breakout_12hTrend_Volume"
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
    
    # 12h data for trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 12h bar
    close_12h_prev = np.roll(close_12h, 1)
    high_12h_prev = np.roll(high_12h, 1)
    low_12h_prev = np.roll(low_12h, 1)
    close_12h_prev[0] = close_12h[0]
    high_12h_prev[0] = high_12h[0]
    low_12h_prev[0] = low_12h[0]
    
    # Camarilla formulas
    R4 = close_12h_prev + (high_12h_prev - low_12h_prev) * 1.1 / 2
    R3 = close_12h_prev + (high_12h_prev - low_12h_prev) * 1.1 / 4
    S3 = close_12h_prev - (high_12h_prev - low_12h_prev) * 1.1 / 4
    S4 = close_12h_prev - (high_12h_prev - low_12h_prev) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_12h, R4)
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    S4_aligned = align_htf_to_ltf(prices, df_12h, S4)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume spike and uptrend
            long_cond = (close[i] > R4_aligned[i] and 
                        ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and
                        volume_spike[i])
            
            # Short breakdown: price breaks below S4 with volume spike and downtrend
            short_cond = (close[i] < S4_aligned[i] and 
                         ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below R3 (mean reversion)
            if close[i] < R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above S3 (mean reversion)
            if close[i] > S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals