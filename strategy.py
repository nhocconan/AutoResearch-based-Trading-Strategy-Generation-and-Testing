#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Spike"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 1d Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    r3 = (high_1d - low_1d) * 1.1 / 6 + close_1d
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 6
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike detection (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for EMA and Camarilla
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R3, price above EMA34, volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume[i] > vol_ma[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # Short: Close < S3, price below EMA34, volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume[i] > vol_ma[i] * 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < S3 or price below EMA34
            if (close[i] < s3_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > R3 or price above EMA34
            if (close[i] > r3_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Camarilla levels provide high-probability reversal/breakout points.
# EMA34 filter ensures trades align with daily trend, reducing counter-trend entries.
# Volume spike (2x average) confirms strong participation in the breakout.
# Works in bull markets (buy breakouts above R3 in uptrend) and bear markets (sell breakdowns below S3 in downtrend).
# Position size 0.25 limits risk while maintaining sufficient trade frequency for 12h timeframe.