#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla R3/S3 levels from 1d as entry points in direction of 1d trend with volume confirmation.
# Long when: price breaks above R3 in 1d uptrend with volume > 1.3x average.
# Short when: price breaks below S3 in 1d downtrend with volume > 1.3x average.
# Uses 1d trend filter: only trade in direction of daily EMA34 trend.
# Works in bull/bear by following trend and using volume to confirm institutional interest.
# Target: 15-25 trades/year per symbol.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    # But standard Camarilla uses:
    # R4 = close + 1.5*(high - low)
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # S4 = close - 1.5*(high - low)
    # We'll use R3 and S3 as breakout levels
    diff_1d = high_1d - low_1d
    r3_1d = close_1d + 1.1 * diff_1d
    s3_1d = close_1d - 1.1 * diff_1d
    
    # Daily trend filter: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_uptrend = close_1d > ema34_1d
    daily_downtrend = close_1d < ema34_1d
    
    # Align daily data to 12h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Volume confirmation: 20-period average
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.3
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + price breaks above R3 + volume
            if daily_up and volume_confirm:
                if close[i] > r3_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: daily downtrend + price breaks below S3 + volume
            elif daily_down and volume_confirm:
                if close[i] < s3_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: trend changes or price falls below R3
            if not daily_up or close[i] < r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend changes or price rises above S3
            if not daily_down or close[i] > s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals