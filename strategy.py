#!/usr/bin/env python3
# 6H_12h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: Camarilla R3/S3 breakout with 12h EMA trend filter and volume confirmation.
# Uses 6h as primary timeframe for balanced trade frequency (12-37/year) and 12h for trend/levels.
# Works in bull/bear via trend filter: only trade long above EMA, short below EMA.
# Volume confirmation reduces false breakouts. Target: 50-150 total trades over 4 years.

name = "6H_12h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
timeframe = "6h"
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
    
    # Get 12h data for Camarilla pivot levels and EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point and Camarilla levels (R3, S3)
    pivot = (high_12h + low_12h + close_12h) / 3
    range_ = high_12h - low_12h
    r3 = pivot + range_ * 1.1
    s3 = pivot - range_ * 1.1
    
    # EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average (strict)
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 + above 12h EMA34 + volume confirmation
            if close[i] > r3_aligned[i] and close[i] > ema34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + below 12h EMA34 + volume confirmation
            elif close[i] < s3_aligned[i] and close[i] < ema34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 12h EMA34 (trend change)
            if close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 12h EMA34 (trend change)
            if close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals