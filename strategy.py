#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_With_Trend
Fade at Camarilla R3/S3 levels when price is in the daily trend (above/below 1d EMA200).
Long at S3 in uptrend, short at R3 in downtrend. Exit when price reaches R4/S4 or reverses to R2/S2.
Designed to capture mean reversion within the trend with clear risk control.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Daily EMA200 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Daily Camarilla levels (based on previous day) ===
    # Calculate pivots using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formula
    R4 = prev_close + (prev_high - prev_low) * 1.5000
    R3 = prev_close + (prev_high - prev_low) * 1.2500
    R2 = prev_close + (prev_high - prev_low) * 1.1666
    R1 = prev_close + (prev_high - prev_low) * 1.0833
    PP = (prev_high + prev_low + prev_close) / 3
    S1 = prev_close - (prev_high - prev_low) * 1.0833
    S2 = prev_close - (prev_high - prev_low) * 1.1666
    S3 = prev_close - (prev_high - prev_low) * 1.2500
    S4 = prev_close - (prev_high - prev_low) * 1.5000
    
    # Align daily levels to 6h timeframe
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    R2_6h = align_htf_to_ltf(prices, df_1d, R2)
    R1_6h = align_htf_to_ltf(prices, df_1d, R1)
    PP_6h = align_htf_to_ltf(prices, df_1d, PP)
    S1_6h = align_htf_to_ltf(prices, df_1d, S1)
    S2_6h = align_htf_to_ltf(prices, df_1d, S2)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or 
            np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or
            np.isnan(R2_6h[i]) or np.isnan(S2_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price at S3, above daily EMA200 (uptrend)
            if (close[i] <= S3_6h[i] * 1.001 and  # Allow small tolerance for touching S3
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price at R3, below daily EMA200 (downtrend)
            elif (close[i] >= R3_6h[i] * 0.999 and  # Allow small tolerance for touching R3
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price reaches R4 (target) or reverses to S2 (stop)
            if (close[i] >= R4_6h[i] * 0.999 or  # Hit target
                close[i] <= S2_6h[i] * 1.001):    # Reverse to S2
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S4 (target) or reverses to R2 (stop)
            if (close[i] <= S4_6h[i] * 1.001 or  # Hit target
                close[i] >= R2_6h[i] * 0.999):    # Reverse to R2
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Fade_With_Trend"
timeframe = "6h"
leverage = 1.0