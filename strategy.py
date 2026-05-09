#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Combining strong breakout signals (Camarilla R3/S3) with 1D trend filter and volume surge reduces false signals while capturing momentum. Works in bull (breakouts continue) and bear (reversals at S3/R3) by requiring volume confirmation and trend alignment. Target: 25-40 trades/year.
"""

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] * 2 + ema34_1d[i-1] * 32) / 34
    
    # Align 1d EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels for each 1d bar: R3, S3
    camarilla_r3_1d = np.full_like(close_1d, np.nan)
    camarilla_s3_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            camarilla_r3_1d[i] = close_1d[i] + 1.1 * (high_1d[i] - low_1d[i]) / 2
            camarilla_s3_1d[i] = close_1d[i] - 1.1 * (high_1d[i] - low_1d[i]) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need 1d EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        trend_up = close[i] > ema34_1d_aligned[i]
        volume_surge = volume_ratio[i] > 1.8
        
        if position == 0:
            # Enter long: Uptrend + price breaks above R3 + volume surge
            if trend_up and close[i] > camarilla_r3_1d_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below S3 + volume surge
            elif not trend_up and close[i] < camarilla_s3_1d_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below S3
            if not trend_up or close[i] < camarilla_s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above R3
            if trend_up or close[i] > camarilla_r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals