#!/usr/bin/env python3
# 6H_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike
# Hypothesis: Breakout of Camarilla R3/S3 levels with 12h trend filter and volume spike.
# Works in bull/bear by using 12h trend for direction and volume to confirm institutional interest.
# Target: 20-30 trades/year per symbol (60-90 total over 4 years).

name = "6H_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
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
    
    # Calculate Camarilla levels for each bar using previous day's range
    camarilla_high = np.zeros(n)
    camarilla_low = np.zeros(n)
    
    for i in range(1, n):
        # Previous bar's high and low (daily approximation)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        if prev_high <= prev_low:
            camarilla_high[i] = camarilla_low[i] = prev_close
        else:
            range_val = prev_high - prev_low
            camarilla_high[i] = prev_close + 1.1 * range_val * 1.1 / 2  # R3 level
            camarilla_low[i] = prev_close - 1.1 * range_val * 1.1 / 2   # S3 level
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ma50_12h = pd.Series(close_12h).rolling(window=50, min_periods=50).mean().values
    trend_12h_up = close_12h > ma50_12h
    trend_12h_down = close_12h < ma50_12h
    
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Volume spike filter (2x 20-period average)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(trend_12h_up_aligned[i]) or
            np.isnan(trend_12h_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        trend_up = trend_12h_up_aligned[i] > 0.5
        trend_down = trend_12h_down_aligned[i] > 0.5
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and 12h uptrend
            if close[i] > camarilla_high[i] and volume_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and 12h downtrend
            elif close[i] < camarilla_low[i] and volume_spike and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below R3 or trend changes
            if close[i] < camarilla_high[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above S3 or trend changes
            if close[i] > camarilla_low[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals