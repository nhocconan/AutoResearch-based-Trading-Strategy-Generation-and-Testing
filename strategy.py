#!/usr/bin/env python3
# 6H_12H_1D_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: On 6h timeframe, enter long when price breaks above Camarilla R3 from daily pivot
# with confirmation from 12h EMA50 (trend filter) and volume spike (>1.5x 20-period avg).
# Enter short when price breaks below S3 with same conditions reversed.
# Uses 12h trend filter to avoid counter-trend trades, works in bull/bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6H_12H_1D_Camarilla_R3_S3_Breakout_12hTrend_Volume"
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
    
    # Get daily data for Camarilla pivot levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels (R3, S3)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r3 = pivot + range_ * 1.1 / 2  # R3 = pivot + (range * 1.1 / 2)
    s3 = pivot - range_ * 1.1 / 2  # S3 = pivot - (range * 1.1 / 2)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 + above 12h EMA50 + volume confirmation
            if close[i] > r3_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + below 12h EMA50 + volume confirmation
            elif close[i] < s3_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 12h EMA50 (trend change)
            if close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 12h EMA50 (trend change)
            if close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals