#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeS
Hypothesis: Use 12h EMA trend direction, Camarilla R3/S3 levels from daily as breakout levels,
with volume confirmation. Designed to capture breakouts in trending markets while avoiding
false signals in ranging conditions. Designed for low trade frequency and high win rate.
"""

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeS"
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
    
    # Get 12h data for trend direction
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA for trend direction
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_prev = np.roll(ema_12h, 1)
    ema_12h_prev[0] = ema_12h[0]
    trend_12h = np.where(ema_12h > ema_12h_prev, 1, -1)
    
    # Align 12h trend to 4h timeframe
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h.astype(float))
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_range = (high_1d - low_1d)
    r3_level = close_1d + 1.1 * camarilla_range / 2
    s3_level = close_1d - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(trend_12h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 12h uptrend AND price breaks above R3 AND volume filter
            if trend_12h_aligned[i] == 1 and close[i] > r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: 12h downtrend AND price breaks below S3 AND volume filter
            elif trend_12h_aligned[i] == -1 and close[i] < s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 12h trend turns down OR price drops below S3 (reversal signal)
            if trend_12h_aligned[i] == -1 or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: 12h trend turns up OR price rises above R3 (reversal signal)
            if trend_12h_aligned[i] == 1 or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals