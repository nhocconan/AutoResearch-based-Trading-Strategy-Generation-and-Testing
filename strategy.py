#!/usr/bin/env python3
# 4H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Trade breakouts of Camarilla R3/S3 levels in direction of daily EMA34 trend with volume confirmation.
# Works in bull/bear by following daily trend and using volume to confirm institutional participation.
# Target: 20-40 trades/year per symbol.

name = "4H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Daily Camarilla levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_range = high_1d - low_1d
    r3_level = close_1d + 1.1 * camarilla_range / 2
    s3_level = close_1d - 1.1 * camarilla_range / 2
    
    # Daily trend filter (EMA34)
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_uptrend = close_1d > ema34_1d
    daily_downtrend = close_1d < ema34_1d
    
    # Align to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # 4h indicators
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(daily_uptrend_aligned[i]) or 
            np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + price breaks above R3 + volume
            if daily_up and close[i] > r3_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: daily downtrend + price breaks below S3 + volume
            elif daily_down and close[i] < s3_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price drops back below R3 or trend changes
            if close[i] < r3_aligned[i] or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above S3 or trend changes
            if close[i] > s3_aligned[i] or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals