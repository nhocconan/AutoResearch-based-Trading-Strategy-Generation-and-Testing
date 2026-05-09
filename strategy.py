#!/usr/bin/env python3
# 6H_1W_1D_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: On 6h timeframe, use weekly trend filter from weekly close vs open and daily trend from EMA34,
# with Camarilla R3/S3 breakouts from daily levels and volume confirmation.
# Weekly trend provides macro bias, daily trend filters intermediate noise, and R3/S3 breakouts capture
# strong moves with validation. Volume spike ensures breakout conviction. Designed to work in both bull
# and bear markets via dual timeframe trend filters (weekly + daily).
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6H_1W_1D_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Get weekly data for trend filter (weekly close vs open)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_bull = close_1w > open_1w  # True if weekly close > open (bullish week)
    
    # Get daily data for Camarilla levels and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels (R3, S3)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r3 = pivot + range_ * 1.1 / 2  # R3 = pivot + (range * 1.1 / 2)
    s3 = pivot - range_ * 1.1 / 2  # S3 = pivot - (range * 1.1 / 2)
    
    # EMA34 for daily trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 2.0)
    
    # Align all to 6h timeframe
    weekly_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_bull)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 + weekly bullish + price above EMA34 + volume confirmation
            if (close[i] > r3_aligned[i] and 
                weekly_bull_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + weekly bearish + price below EMA34 + volume confirmation
            elif (close[i] < s3_aligned[i] and 
                  not weekly_bull_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below EMA34 (trend change) or weekly bearish flip
            if close[i] < ema34_1d_aligned[i] or not weekly_bull_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above EMA34 (trend change) or weekly bullish flip
            if close[i] > ema34_1d_aligned[i] or weekly_bull_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals