#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla pivot reversal with 12-hour trend filter.
Long at S1 support when price bounces up from S1 and 12h EMA50 is rising.
Short at R1 resistance when price rejects down from R1 and 12h EMA50 is falling.
Exit when price crosses the pivot point (PP) or reaches opposite S3/R3 level.
Camarilla levels provide high-probability reversal zones in ranging markets.
Trend filter ensures we only trade with the higher timeframe momentum.
Works in both bull and bear markets by adapting to 12h trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Camarilla pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas: PP = (H+L+C)/3, Range = H-L
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Key levels: S1, S2, S3, R1, R2, R3
    s1 = close_1d - (range_1d * 1.0 / 6.0)
    s2 = close_1d - (range_1d * 2.0 / 6.0)
    s3 = close_1d - (range_1d * 3.0 / 6.0)
    r1 = close_1d + (range_1d * 1.0 / 6.0)
    r2 = close_1d + (range_1d * 2.0 / 6.0)
    r3 = close_1d + (range_1d * 3.0 / 6.0)
    pp_val = pp  # pivot point
    
    # Align Camarilla levels to 4h timeframe (1-day levels shift only at daily boundary)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_val)
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend direction
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation - 4h volume vs 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: Price at S1 support with bullish 12h trend and volume
            if (close[i] <= s1_aligned[i] * 1.001 and  # Allow small tolerance
                close[i] > s1_aligned[i] and           # Must be above S1
                ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and  # 12h EMA rising
                volume[i] > vol_ma_20[i]):             # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short setup: Price at R1 resistance with bearish 12h trend and volume
            elif (close[i] >= r1_aligned[i] * 0.999 and  # Allow small tolerance
                  close[i] < r1_aligned[i] and           # Must be below R1
                  ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and  # 12h EMA falling
                  volume[i] > vol_ma_20[i]):             # Volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses PP (take profit) or hits S3 (stop)
                if close[i] >= pp_aligned[i] or close[i] <= s3_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses PP (take profit) or hits R3 (stop)
                if close[i] <= pp_aligned[i] or close[i] >= r3_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_S1R1_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0