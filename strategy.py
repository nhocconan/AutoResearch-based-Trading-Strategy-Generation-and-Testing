#!/usr/bin/env python3
"""
6h_WeeklyPivot_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Use weekly pivot points (R3/S3) on 6s timeframe with 1d EMA34 trend filter and volume spike confirmation.
Goes long when price breaks above weekly R3 in uptrend, short when breaks below weekly S3 in downtrend.
Weekly pivots provide stronger support/resistance than daily, reducing false breakouts.
Volume spike (>2x 20-bar MA) confirms institutional interest. Designed for low trade frequency
(12-37 trades/year) to minimize fee drag while capturing strong directional moves in both bull and bear markets.
"""

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
    
    # Get weekly data for pivot points (using Friday's close as weekly close)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Standard formula: PP = (H + L + C) / 3
    # R3 = H + 2*(PP - L) = 3*H - 2*L
    # S3 = L - 2*(H - PP) = 3*L - 2*H
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly R3 and S3
    weekly_r3 = 3 * high_w - 2 * low_w
    weekly_s3 = 3 * low_w - 2 * high_w
    
    # Align weekly levels to 6h timeframe
    weekly_r3_aligned = align_htf_to_ltf(prices, df_w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_w, weekly_s3)
    
    # Volume confirmation: >2x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(weekly_s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions
        breakout_r3 = close[i] > weekly_r3_aligned[i]
        breakdown_s3 = close[i] < weekly_s3_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Entry logic: breakout in direction of trend with volume
        long_entry = vol_confirm and uptrend and breakout_r3
        short_entry = vol_confirm and downtrend and breakdown_s3
        
        # Exit logic: opposite breakout or trend change
        long_exit = breakdown_s3 or (not uptrend)
        short_exit = breakout_r3 or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0