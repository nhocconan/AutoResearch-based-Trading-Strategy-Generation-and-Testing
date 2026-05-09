# 12h_Camarilla_Breakout_1dTrend_Volume_Slow
# Hypothesis: Using 12h timeframe reduces trade frequency to avoid fee drag.
# Combines 1d Camarilla R3/S3 breakouts with 1d EMA trend filter and volume surge.
# Designed for fewer trades (target 12-37/year) with strong confirmation to work in both bull and bear markets.
# Uses discrete position sizing (0.25) to minimize churn and manage drawdown.

name = "12h_Camarilla_Breakout_1dTrend_Volume_Slow"
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
    
    # Get 1d data for Camarilla levels, trend filter, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] * 2 + ema34_1d[i-1] * 32) / 34
    
    # Calculate 1d EMA34 for volume average (same period)
    vol_ema34_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 34:
        vol_ema34_1d[33] = np.mean(volume_1d[0:34])
        for i in range(34, len(volume_1d)):
            vol_ema34_1d[i] = (volume_1d[i] * 2 + vol_ema34_1d[i-1] * 32) / 34
    
    # Align 1d indicators to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema34_1d)
    
    # Calculate Camarilla levels for each 1d bar: R3, S3
    camarilla_r3_1d = np.full_like(close_1d, np.nan)
    camarilla_s3_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            camarilla_r3_1d[i] = close_1d[i] + 1.1 * (high_1d[i] - low_1d[i]) / 2
            camarilla_s3_1d[i] = close_1d[i] - 1.1 * (high_1d[i] - low_1d[i]) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA34 and volume average
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ema34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        trend_up = close[i] > ema34_1d_aligned[i]
        volume_surge = volume[i] > vol_ema34_1d_aligned[i] * 2.0  # Require 2x average volume
        
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

#!/usr/bin/env python3