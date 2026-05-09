#!/usr/bin/env python3
# 1D_Weekly_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Weekly Camarilla R3/S3 breakout with daily trend and volume confirmation on daily timeframe.
# Uses weekly pivot levels for structure, daily EMA34 for trend filter, and volume spike for confirmation.
# Designed to work in both bull and bear markets by following weekly structure and avoiding counter-trend trades.
# Target: 10-30 trades per year (~40-120 over 4 years) to minimize fee drag.

name = "1D_Weekly_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point and Camarilla levels (R3, S3)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r3_1w = pivot_1w + range_1w * 1.1
    s3_1w = pivot_1w - range_1w * 1.1
    
    # Daily EMA34 for trend filter
    ema34_d = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-day average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 2.0)
    
    # Align weekly levels to daily timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for EMA34
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if weekly data not ready
        if np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly R3 + above daily EMA34 + volume confirmation
            if close[i] > r3_1w_aligned[i] and close[i] > ema34_d[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S3 + below daily EMA34 + volume confirmation
            elif close[i] < s3_1w_aligned[i] and close[i] < ema34_d[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below daily EMA34 (trend change)
            if close[i] < ema34_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above daily EMA34 (trend change)
            if close[i] > ema34_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals