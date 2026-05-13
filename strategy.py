#!/usr/bin/env python3
"""
12h_Adaptive_Camarilla_R3_S3_Exit
Hypothesis: Uses 1-day Camarilla R3/S3 levels for breakout entries, filtered by 1-week trend via EMA50, with volume spike confirmation. Exits when price crosses the 1-day EMA34. Designed for 12h timeframe to capture major moves with low trade frequency (target 10-30/year), reducing fee impact while adapting to bull/bear regimes via trend filter.
"""

name = "12h_Adaptive_Camarilla_R3_S3_Exit"
timeframe = "12h"
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
    
    # Get 1d data for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Calculate 1d EMA34 for exit
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    r3_level = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_level = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume average (10-period) for volume spike filter
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_level[i]) or np.isnan(s3_level[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_10[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 10-period average
        vol_spike = volume[i] > 1.5 * vol_ma_10[i]
        
        if position == 0:
            # LONG: Price breaks above R3 + volume spike + weekly uptrend
            if close[i] > r3_level[i] and vol_spike and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below S3 + volume spike + weekly downtrend
            elif close[i] < s3_level[i] and vol_spike and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA34
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA34
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals